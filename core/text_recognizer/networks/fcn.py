"""PyTorch network code for the fully-convolutional network used for line detection."""
from typing import List, Tuple
import torch
from torch import nn

def cal_padding(kernel_size, dilation_rate):
    k, d = kernel_size, dilation_rate
    padding = int( ((k-1)*(d-1)-1+k)/2 )
    return padding

class ResidualConvBlock(nn.Module):
    """Class to instantiate a Residual convolutional block."""
    def __init__(self, input_channel, kernel_sizes, num_filters, dilation_rates):
        super(ResidualConvBlock, self).__init__()
        # calculate paddings to ensure same size
        self.x0 = nn.Sequential(
            nn.Conv2d(input_channel, num_filters[0], kernel_sizes[0], dilation=dilation_rates[0], padding=cal_padding(kernel_sizes[0], dilation_rates[0])),
            nn.ReLU()
        )
        self.x1 = nn.Conv2d(num_filters[0], num_filters[1], kernel_sizes[1], dilation=dilation_rates[1], padding=cal_padding(kernel_sizes[1], dilation_rates[1]))
        self.y = nn.Conv2d(input_channel, num_filters[1], kernel_size=1, padding=cal_padding(1,1))
    
    def forward(self, x):
        return nn.functional.relu(self.x1(self.x0(x)) + self.y(x))


def fcn(_input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
    """Function to instantiate a fully convolutional residual network for line detection."""
    # very simple structure, several residual blocks + final upscale by 1 Conv2D, there's no fusion between dif layers
    i_channel = 1
    num_filters = [8] * 14
    kernel_sizes = [7] * 14
    dilation_rates = [3] * 4 + [7] * 10

    num_classes = output_shape[-1]
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            layers = []
            for i in range(0, len(num_filters), 2):
                if i == 0: input_channel = i_channel
                else: input_channel = num_filters[i-1]
                layers.append(ResidualConvBlock(input_channel, kernel_sizes[i:i+2], num_filters[i:i+2], dilation_rates[i:i+2]))

            layers.append(nn.Conv2d(num_filters[-1], num_classes, kernel_size=1, stride=1, padding=cal_padding(1,1), dilation=1))
            layers.append(nn.LogSoftmax(dim=1)) # explicitly compute prob here b.c. later will be used in prediction
            self.layers = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.layers(x.unsqueeze(1))

    model = Model()
    return model
    