"""Define mlp network function."""
from typing import Tuple

from torch import nn

def mlp(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: int = 128,
        dropout_amount: float = 0.2,
        num_layers: int = 3) -> nn.Module:
    """
    Simple multi-layer perceptron: just fully-connected layers with dropout between them, with softmax predictions.
    Creates num_layers layers.
    """
    num_classes = output_shape[0]
    def flatten(array):
        num = 1
        for i in array:
            num *= i
        return num

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            layers = []
            for i in range(num_layers):
                if i == 0:
                    layers.append(nn.Linear(flatten(input_shape), layer_size))
                else:
                    layers.append(nn.Linear(layer_size, layer_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_amount))
            layers.append(nn.Linear(layer_size, num_classes))
            layers.append(nn.LogSoftmax(dim=1))
            # NOTE need Softmax() for BCELoss, but not for CrossEntropyLoss
            self.layers = nn.Sequential(*layers)
        
        def forward(self, x):
            x = x.view(x.shape[0], -1)
            output = self.layers(x)
            return output

    model = MLP()

    return model
