"""CNN-based model for recognizing handwritten text."""
from typing import Tuple

import torch
from torch import nn


def line_cnn_all_conv(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    window_width: float = 16,
    window_stride: float = 8,
) -> nn.Module:
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    class LineCNN(nn.Module):
        def __init__(self):
            super(LineCNN, self).__init__()
            # the head of LeNet, before Flatten and Dense
            self.l1 = nn.Sequential(
                *[
                    nn.Conv2d(1, 32, 3),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout(0.2),
                ]
            )

            new_height = image_height // 2 - 2
            new_width = image_width // 2 - 2
            new_window_width = window_width // 2 - 2
            new_window_stride = window_stride // 2
            self.l2 = nn.Sequential(
                *[
                    nn.Conv2d(
                        64, 128, (new_height, new_window_width), (1, new_window_stride)
                    ),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )  # (128, 1, num_windows)

            # count how many windows contain outputs
            self.num_windows = (
                int((new_width - new_window_width) / new_window_stride) + 1
            )

            # need to reshape to (1, num_windows, 128) before feeding
            # note the following input channel is 1
            width = int(self.num_windows / output_length)
            self.l3 = nn.Sequential(
                *[
                    nn.Conv2d(1, num_classes, (width, 128), (width, 1)),
                    nn.LogSoftmax(dim=1),
                ]
            )

        def forward(self, x):
            x = torch.unsqueeze(x, dim=1)
            x = self.l1(x)
            x = self.l2(x)
            x = x.permute(0, 2, 3, 1)
            x = self.l3(x)
            x = torch.squeeze(x, dim=3)  # (batch, num_classes, image_width/width, 1)
            x = x[:, :, :output_length]
            return x

    model = LineCNN()

    return model
