from torch.nn import (Module, Conv2d, Linear, MaxPool2d, ReLU, BatchNorm2d, Dropout2d, Dropout, Flatten)

from torch import flatten



class CNNBasic(Module):
    def __init__(self, numChannels=1, classes=1):
        # call the parent constructor
        super(CNNBasic, self).__init__()

        self.conv1 = ConvBlock(in_channels=numChannels, out_channels=12)
        self.conv2 = ConvBlock(in_channels=12, out_channels=24)
        self.conv3 = ConvBlock(in_channels=24, out_channels=48, kernel_size=(5, 5))
        self.conv4 = ConvBlock(in_channels=48, out_channels=96, kernel_size=(5, 5))
        self.conv5 = ConvBlock(in_channels=96, out_channels=192, kernel_size=(5, 5))
        self.conv6 = ConvBlock(in_channels=192, out_channels=384, kernel_size=(5, 5))


        self.fc1 = Linear(in_features=6144, out_features=500)
        self.relu5 = ReLU()

        self.fc2 = Linear(in_features=500, out_features=classes)
        
    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)

        x = self.conv6(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu5(x)
        output = self.fc2(x)
        # return the output predictions
        return output
    

class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        # call the parent constructor
        super(ConvBlock, self).__init__()
        # initialize the convolutional layer
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size)
        # initialize the batch normalization layer
        self.bn = BatchNorm2d(num_features=out_channels)
        # initialize the ReLU layer
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        # apply the convolutional layer, followed by batch normalization
        # and relu activation
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # return the block
        return x