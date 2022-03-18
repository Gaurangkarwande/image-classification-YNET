from torch import nn
from torch.nn import functional as F

class ConvLayer(nn.Module):
    def __init__(self, config, is_left=True) -> None:
        super(ConvLayer, self).__init__()
        self.dilation = 1 if is_left else 2
        self.padding = 0 if is_left else 2 
        self.conv1 = nn.Conv2d(in_channels=config['cnn1_in'], out_channels=config['cnn2_in'], \
                                kernel_size=config['kernel_size'], dilation=self.dilation)
        self.batchnorm1 = nn.BatchNorm2d(config['cnn2_in'])
        self.conv2 = nn.Conv2d(in_channels=config['cnn2_in'], out_channels=config['cnn3_in'], kernel_size=config['kernel_size'], dilation=self.dilation)
        self.batchnorm2 = nn.BatchNorm2d(config['cnn3_in'])
        self.conv3 = nn.Conv2d(in_channels=config['cnn3_in'], out_channels=config['cnn3_out'], kernel_size=config['kernel_size'], dilation=self.dilation, padding=self.padding)
        self.batchnorm3 = nn.BatchNorm2d(config['cnn3_out'])
        self.pooling = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        x = self.conv1(x)           # left: N x 32 x 30 x 30, right: N x 32 x 28 x 28
        x = self.batchnorm1(x)
        x = self.pooling(x)
        x = F.leaky_relu(x)

        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.pooling(x)
        x = F.leaky_relu(x)

        

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.pooling(x)
        x = F.leaky_relu(x)        

        return x                    # N x 128 x 2 x 2
