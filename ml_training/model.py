from torch import nn
from torch.nn import Module



class GRU(Module):
    def __init__(self, sequence_len=200, input_size=3, hidden_size=32, num_layers=1, classes=1):
        # call the parent constructor
        super(GRU, self).__init__()

        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
        # No activation function is used because we are using BCEWithLogitsLoss
        
    def forward(self, x, h0: list):
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
    
    