from torch import nn
import torch
import math
from torch.nn import Module
from local_attention import LocalAttention


class GRU(Module):
    def __init__(self, sequence_len=200, input_size=3, hidden_size=32, num_layers=1, classes=1, use_attention=True, context_length=20):
        # call the parent constructor
        super(GRU, self).__init__()

        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.context_length = context_length
        self.use_attention = use_attention

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

        if self.use_attention:
            self.local_attention = LocalAttention(dim=hidden_size*2,
                                            window_size=20, 
                                            causal=False, 
                                            look_forward=1, 
                                            look_backward=0,
                                            dropout=0.1,
                                            exact_windowsize=True,
                                            autopad=True)

        self.fc = nn.Linear(hidden_size*2, classes)
        # No activation function is used because we are using BCEWithLogitsLoss

        # We need to create the default positional embeddings
        #self.positional_embeddings = 

        
    def forward(self, x, h0: list = []):
        if not h0:
            h0 = [None]*self.num_layers
        for i in range(self.num_layers):
            x, _ = self.grus[i](x, h0[i])
        #x = self.positional_embeddings(x)
        if self.use_attention:
            x = self.local_attention(x, x, x)
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
    
    def gen_pe(max_length, d_model, n):
        # Taken from: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
        # calculate the div_term
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(n) / d_model))
        # generate the positions into a column matrix
        k = torch.arange(0, max_length).unsqueeze(1)
        # generate an empty tensor
        pe = torch.zeros(max_length, d_model)
        # set the even values
        pe[:, 0::2] = torch.sin(k * div_term)
        # set the odd values
        pe[:, 1::2] = torch.cos(k * div_term)
        # add a dimension       
        pe = pe.unsqueeze(0)
        # the output has a shape of (1, max_length, d_model)
        return pe    
