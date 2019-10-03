"""LSTM with CTC for handwritten text recognition within a line."""
# from tensorflow.python.client import device_lib  # pylint: disable=no-name-in-module
# from tensorflow.keras.layers import Dense, Input, Reshape, TimeDistributed, Lambda, LSTM, CuDNNLSTM
# from tensorflow.keras.models import Model as KerasModel

import torch
from torch import nn

from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window
# from text_recognizer.networks.ctc import ctc_decode

def line_lstm_ctc(input_shape, output_shape, window_width=28, window_stride=14):  # pylint: disable=too-many-locals
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    num_windows = int((image_width - window_width) / window_stride) + 1
    if num_windows < output_length:
        raise ValueError(f'Window width/stride need to generate >= {output_length} windows (currently {num_windows})')

    convnet = lenet((image_height, window_width), (num_classes,))
    time_steps = int((image_width-window_width) / window_stride) + 1

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
            conv_out = torch.stack([self.conv2(self.conv1(patches[:,:,:,:,i]).view(B, -1)) for i in range(T)], dim=0) # (T, B, 128)
            lstm_out, (h_n, c_n) = self.lstm(conv_out) # lstm_out: (T, B, 128)
            x_linear = lstm_out.view(T * B, 128)
            out_linear = self.linear(x_linear).view(T, B, num_classes)
            logsoftmax = nn.functional.log_softmax(out_linear, dim=2) # logsoftmax should be in shape (T, B, classes)
            input_lengths = (torch.ones(B) * T).to(torch.int)

            return logsoftmax, input_lengths
    
    model = ModelCTC()

    return model
