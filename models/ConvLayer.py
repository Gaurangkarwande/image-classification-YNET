from torch import nn

class ConvLayer(nn.Module):
    def __init__(self, config, is_left=True) -> None:
        super(ConvLayer, self).__init__()
        self.dilation = 1 if is_left else 2
        self.padding = 0 if is_left else 2 
        self.conv1 = nn.Conv2d(in_channels=config['cnn1_in'], out_channels=config['cnn2_in'], \
                                kernel_size=config['kernel_size'], dilation=self.dilation)
        self.conv2 = nn.Conv2d(in_channels=config['cnn2_in'], out_channels=config['cnn3_in'], kernel_size=config['kernel_size'], dilation=self.dilation)
        self.conv3 = nn.Conv2d(in_channels=config['cnn3_in'], out_channels=config['cnn3_out'], kernel_size=config['kernel_size'], dilation=self.dilation, padding=self.padding)
        self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)           # left: N x 32 x 30 x 30, right: N x 32 x 28 x 28
        x = self.pooling(x)
        x = x.relu()

        x = self.conv2(x)
        x = self.pooling(x)
        x = x.relu()

        x = self.conv3(x)
        x = self.pooling(x)
        x = x.relu()        

        return x                    # N x 128 x 2 x 2
