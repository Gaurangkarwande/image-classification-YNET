import torch
from torch import nn
from models.ConvLayer import ConvLayer


class YNet(nn.Module):
    def __init__(self, config):
        super(YNet, self).__init__()
        self.cnn_left = ConvLayer(config, is_left=True)
        self.cnn_right = ConvLayer(config, is_left=False)
        self.classifier = nn.Sequential(nn.Dropout(config['dropout']), nn.Linear(in_features=config['linear_in'], out_features=config['n_classes']))
    
    def forward(self, x, infer=False):
        x_left = self.cnn_left(x)       # batch x 128 x 2 x 2
        x_right = self.cnn_right(x)     # batch x 128 x 2 x 2
        x_full = torch.concat((x_left, x_right), dim=1)     # batch x 256 x 2 x 2
        x_full = x_full.view(x_full.shape[0], -1)
        output = self.classifier(x_full)
        if infer:
            return output, x_left, x_right, x_full
        del x_left; del x_right; x_full; del x
        return output