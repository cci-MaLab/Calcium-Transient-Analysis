from torch import nn
from torch.nn import Module



class GRU(Module):
    def __init__(self, sequence_len=200, input_size=3, hidden_size=32, num_layers=1, classes=1, context_length=20):
        # call the parent constructor
        super(GRU, self).__init__()

        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.context_length = context_length

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

        self.attn_forward = SelfAttention(hidden_size*2)
        self.attn_backward = SelfAttention(hidden_size*2)
        self.fc = nn.Linear(hidden_size*2, classes)
        # No activation function is used because we are using BCEWithLogitsLoss
        
    def forward(self, x, h0: list = []):
        if not h0:
            h0 = [None]*self.num_layers
        for i in range(self.num_layers):
            x, _ = self.grus[i](x, h0[i])
        # Iterate through every time step and apply attention
        x_new = []
        for i in range(self.sequence_len):
            lower = nn.max(0, i - self.context_length)
            upper = nn.min(self.sequence_len, i + self.context_length+1)
            target = x[:, i, :]
            forward_context = x[:, lower:i+1, :]
            backward_context = x[:, i:upper, :]
            forward = self.attn_forward(target, forward_context)
            backward = self.attn_backward(target, backward_context)
            x_new.append(nn.cat([forward, backward], dim=1))
        x = nn.stack(x_new, dim=1)

        
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


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, target, context):
        queries = self.query(context)
        keys = self.key(target)
        values = self.value(context)
        scores = nn.mm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = nn.mm(attention, values)
        return weighted