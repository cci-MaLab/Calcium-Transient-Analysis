from torch import nn
import torch
import math
from torch.nn import Module, TransformerEncoderLayer, TransformerEncoder
from local_attention import LocalAttention
from local_attention.transformer import LocalMHA, FeedForward, DynamicPositionBias, eval_decorator, exists, rearrange, top_k
import torch.nn.functional as F
from ml_training import config

class GRU(Module):
    def __init__(self, sequence_len=200, slack=50, inputs=["YrA", "C", "DFF"], hidden_size=32, num_layers=1, classes=1):
        # call the parent constructor
        super(GRU, self).__init__()

        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.slack = slack
        input_size = len(inputs)

        '''
        We cannot use the num_layers because Pytorch only provides the output for the last layer. The reason why we need
        to the intermediary layer is because of the mini-epoch approach that requires us to preserve those hidden states
        in each layer. We will have to manually create into a list the GRU layers.
        '''
        self.gru = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size*2, classes)

        
    def forward(self, x):
        x, _ = self.gru(x)

        x = self.fc(x)
        return torch.squeeze(x[:,self.slack:-self.slack,:], dim=-1)
    
class LSTM(Module):
    def __init__(self, sequence_len=200, slack=50, inputs=["YrA", "C", "DFF"], hidden_size=32, num_layers=1, classes=1):
        # call the parent constructor
        super(LSTM, self).__init__()

        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.slack = slack
        input_size = len(inputs)

        '''
        We cannot use the num_layers because Pytorch only provides the output for the last layer. The reason why we need
        to the intermediary layer is because of the mini-epoch approach that requires us to preserve those hidden states
        in each layer. We will have to manually create into a list the GRU layers.
        '''
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size*2, classes)

        
    def forward(self, x):
        x, _ = self.lstm(x)

        x = self.fc(x)
        return torch.squeeze(x[:,self.slack:-self.slack,:], dim=-1)

class LocalTransformer(nn.Module):
    """ 
    Taken from: https://github.com/lucidrains/local-attention
    Adjusted slightly to work with numerical non-tokenized data
    """
    def __init__(
        self,
        *,
        max_seq_len,
        inputs=["C", "DFF"],
        depth=1,
        causal = True,
        local_attn_window_size = 10,
        dim_head = 32,
        heads = 1,
        ff_mult = 2,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_xpos = False,
        xpos_scale_base = None,
        use_dynamic_pos_bias = False,
        slack = 50,
        sequence_len = 200,
        linear_expansion_dim = 32,
        **kwargs
    ):
        super().__init__()
        self.inputs = inputs
        dim = len(inputs)
        self.linear_expansion = nn.Linear(dim, linear_expansion_dim)
        self.pos_emb = nn.Embedding(max_seq_len, linear_expansion_dim)
        self.slack = slack
        self.sequence_len = sequence_len
        

        self.max_seq_len = max_seq_len
        self.layers = nn.ModuleList([])

        self.local_attn_window_size = local_attn_window_size
        self.dynamic_pos_bias = None
        if use_dynamic_pos_bias:
            self.dynamic_pos_bias = DynamicPositionBias(dim = linear_expansion_dim // 2, heads = heads)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LocalMHA(dim = linear_expansion_dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, causal = causal,
                          window_size = local_attn_window_size, use_xpos = use_xpos, xpos_scale_base = xpos_scale_base,
                            use_rotary_pos_emb = not use_dynamic_pos_bias, prenorm = True, **kwargs),
                FeedForward(dim = linear_expansion_dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(linear_expansion_dim),
            nn.Linear(linear_expansion_dim, 1, bias = False)
        )

    def forward(self, x, mask = None):
        n, device = x.shape[1], x.device

        assert n <= self.max_seq_len
        x = self.linear_expansion(x)
        x = x + self.pos_emb(torch.arange(n, device = device))

        # dynamic pos bias
        attn_bias = None
        if exists(self.dynamic_pos_bias):
            w = self.local_attn_window_size
            attn_bias = self.dynamic_pos_bias(w, w * 2)

        # go through layers
        for attn, ff in self.layers:
            x = attn(x, mask = mask, attn_bias = attn_bias) + x
            x = ff(x) + x

        logits = self.to_logits(x)

        return torch.squeeze(logits[:, self.slack:-self.slack,:], dim=-1)
    

class BasicTransformer(Module):
    def __init__(self, sequence_len=200, slack=50, inputs=["YrA", "C", "DFF"], hidden_size=32, num_layers=1, num_heads=1, classes=1):
        # call the parent constructor
        super(BasicTransformer, self).__init__()

        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.slack = slack
        self.inputs = inputs
        input_size = len(inputs)

        self.linear_expansion = nn.Linear(input_size, hidden_size)

        self.pos_emb = pos_enc(sequence_len+2*slack, hidden_size).to(config.DEVICE)

        self.transformer_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size, batch_first=True)
        self.transformer = TransformerEncoder(self.transformer_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, classes)

    def forward(self, x):
        x = self.linear_expansion(x)

        x = x + self.pos_emb

        x = self.transformer(x)

        x = self.fc(x)
        return torch.squeeze(x[:,self.slack:-self.slack,:], -1)
    

def pos_enc(length, dim):
    assert (dim % 2 == 0), "Dimension of positional encoding should be even"
    encoding = torch.zeros(length, dim)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)

    return encoding


class GRU_Hidden(Module):
    def __init__(self, sequence_len=200, inputs=["YrA", "C", "DFF"], hidden_size=32, num_layers=1, classes=1):
        # call the parent constructor
        super(GRU_Hidden, self).__init__()

        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.inputs = inputs
        input_size = len(inputs)

        '''
        We cannot use the num_layers because Pytorch only provides the output for the last layer. The reason why we need
        to the intermediary layer is because of the mini-epoch approach that requires us to preserve those hidden states
        in each layer. We will have to manually create into a list the GRU layers.
        '''
        self.grus = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                self.grus.append(nn.GRU(input_size=input_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True))
            else:
                self.grus.append(nn.GRU(input_size=hidden_size*2, hidden_size=self.hidden_size, bidirectional=True, batch_first=True))

        self.fc = nn.Linear(hidden_size*2, classes)


        
    def forward(self, x, h0: list = []):
        if not h0:
            h0 = [None]*self.num_layers
        for i in range(self.num_layers):
            x, _ = self.grus[i](x, h0[i])

        x = self.fc(x)
        return x
    
    def forward_hidden(self, x):
        # This assumes that the data is unbatched
        h0s = []
        length = x.shape[0]
        for i in range(self.num_layers):           
            x, _ = self.grus[i](x)
            h0s.append(x.view(length, 2, self.hidden_size))

        return h0s
