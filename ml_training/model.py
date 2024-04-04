from torch import nn
from torch.nn import Module



class GRU(Module):
    def __init__(self, sequence_len=200, input_size=3, hidden_size=32, classes=1):
        # call the parent constructor
        super(GRU, self).__init__()

        self.sequence_len = sequence_len
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(64, classes)
        # No activation function is used because we are using BCEWithLogitsLoss
        
    def forward(self, x, h0=None):
        x, _ = self.gru(x, h0)
        x = self.fc(x)
        return x
    
    def forward_hidden(self, x):
        # This assumes that the data is unbatched
        length = x.shape[0]
        x, _ = self.gru(x)

        return x.view(length, 2, self.hidden_size)
    
    