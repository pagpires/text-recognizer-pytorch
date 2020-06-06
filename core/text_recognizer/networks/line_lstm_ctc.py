"""LSTM with CTC for handwritten text recognition within a line."""
import torch
from torch import nn
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window


def line_lstm_ctc(input_shape, output_shape, window_width=28, window_stride=14):
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    num_windows = int((image_width - window_width) / window_stride) + 1
    if num_windows < output_length:
        raise ValueError(
            f"Window width/stride need to generate >= {output_length} windows (currently {num_windows})"
        )

    convnet = lenet((image_height, window_width), (num_classes,))

    class ModelCTC(nn.Module):
        """
        extract image for each window -> conv -> lstm -> dense -> softmax
        """

        def __init__(self):
            super(ModelCTC, self).__init__()
            self.conv1 = convnet.conv_layer
            self.conv2 = convnet.mlp_layer[:-3]
            self.lstm = nn.LSTM(128, 128)
            self.linear = nn.Linear(128, num_classes)

        def forward(self, x):
            x = torch.unsqueeze(x, dim=1)
            patches = slide_window(x, window_width, window_stride)
            B, C, H, Window_W, T = patches.shape

            patches = patches.permute((4, 0, 1, 2, 3))
            # PyTorch's way of TimeDistributed: merge dims T and B
            conv_o1 = self.conv1(
                patches.contiguous().view(T * B, C, H, Window_W)
            )  # (T*B, C, H/2-2, W/2-2)
            conv_out = self.conv2(conv_o1.view(T * B, -1)).view(T, B, 128)

            lstm_out, (h_n, c_n) = self.lstm(conv_out)  # lstm_out: (T, B, 128)
            out_linear = self.linear(lstm_out)  # nn.Linear() allows 3D tensor
            logsoftmax = nn.functional.log_softmax(
                out_linear, dim=2
            )  # logsoftmax should be in shape (T, B, classes) to be consistent with ctc_decode
            input_lengths = torch.Tensor([T] * B).long()

            return logsoftmax, input_lengths

    model = ModelCTC()

    return model
