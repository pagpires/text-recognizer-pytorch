"""LSTM with CTC for handwritten text recognition within a line."""
import torch
from torch import nn
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window

def line_crnn(input_shape, output_shape):
    '''
    CRNN model that shrinks the H dim to 1, and use W dim as the time step,
    thus avoid manual generation of patches with slide_window
    '''
    class BidirectionalLSTM(nn.Module):

        def __init__(self, nIn, nHidden, nOut):
            super(BidirectionalLSTM, self).__init__()

            self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
            self.embedding = nn.Linear(nHidden * 2, nOut)

        def forward(self, x):
            recurrent, _ = self.rnn(x)
            T, b, h = recurrent.size()
            t_rec = recurrent.view(T * b, h)

            output = self.embedding(t_rec)  # (T*B, nOut)
            output = output.view(T, b, -1)

            return output


    class CRNN(nn.Module):

        def __init__(self, imgH, nc, nclass, nh ):
            super(CRNN, self).__init__()
            # (B, 1, 28, 952)
            cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), 
                nn.ReLU(),
                nn.MaxPool2d(2,2), # (B, 64, 14, 476)
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2,2), # (B, 128, 7, 238)
                nn.Conv2d(128, 128, 3, 1), # (B, 128, 5, 236)
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 3, 1), # (B, 128, 3, 234)
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 1), # (B, 128, 1, 232)
                nn.ReLU(),
            )

            self.cnn = cnn
            self.rnn = BidirectionalLSTM(nh, nh, nclass)

        def forward(self, x):
            # conv features
            if len(x.shape) < 4:
                x = x.unsqueeze(1) # ensure a channel dim
            conv = self.cnn(x)
            # _, _, h, _ = conv.size()
            # assert h == 1, f"the height of conv must be 1, but get {h}"
            conv = conv.squeeze(2)
            conv = conv.permute(2, 0, 1)  # (w, b, c)
            w, b, c = conv.shape

            # rnn features
            output = self.rnn(conv) # (w, b, n_class)
            input_lengths = torch.full((b, ), w)

            return output, input_lengths
    
    H, W = input_shape
    output_lengths, nclass = output_shape
    model = CRNN(imgH=H, nc=1, nclass=nclass, nh=128)

    return model
    