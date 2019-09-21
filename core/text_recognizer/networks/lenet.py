"""LeNet network."""
from typing import Tuple
from torch import nn

def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
    """Return LeNet Keras model."""
    num_classes = output_shape[0]
    input_h = input_shape[0]

    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            layers = []
            layers.append(nn.Conv2d(1, 32, 3))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(32, 64, 3))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Dropout(0.2))
            self.conv_layer = nn.Sequential(*layers)
            layers = []
            self.output_h = (input_h-4)//2
            layers.append(nn.Linear(64*self.output_h*self.output_h, 128))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            layers.append(nn.Linear(128, num_classes))
            layers.append(nn.Softmax(dim=1))
            self.mlp_layer = nn.Sequential(*layers)
        
        def forward(self, x):
            if len(x.shape) < 4:
                x = x.unsqueeze(dim=1)
            x = self.conv_layer(x)
            x = x.view(-1, 64 * self.output_h * self.output_h)
            x = self.mlp_layer(x)
            return x

    model = LeNet()

    return model
