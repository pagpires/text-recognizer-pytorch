"""DatasetSequence class."""
import numpy as np
import torch
from torch.utils import data

# Create a Dataset that takes EMNIST's self.X, self.y
# https://www.tinymind.com/learn/terms/hdf5#pytorch
# complex version: https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
# Dataset detail https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class CustomDataset(data.Dataset):
    """ Custom Dataset Wrapper"""
    def __init__(self, x, y, augment_fn=None, format_fn=None):
        self.x = x
        self.y = y
        self.augment_fn = augment_fn
        self.format_fn = format_fn

    def __len__(self):
        """Return length of the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx, :]
        y = np.argmax(self.y[idx, :], axis=-1).astype(np.uint8)

        if self.augment_fn:
            # NOTE read and transformed e2e by transforms
            # do not include ToTensor otherwise y will be rescaled
            # keep io in numpy array
            x, y = self.augment_fn(x, y)
            x = np.array(x)
            y = np.array(y)
            
        x = torch.from_numpy(x)
        # TODO decide whether to keep one-hot-encoding or scalar class, or a dataLoader for each dataset

        if x.dtype == torch.uint8:
            # NOTE should tensor.to(float) before division
            x = (x.to(torch.float32) / 255)
        y = torch.from_numpy(y).long()

        # TODO: 
        if self.format_fn:
            x, y = self.format_fn(x, y)

        return x, y

class DatasetSequence(data.DataLoader):
    def __init__(self, x, y, batch_size=32, augment_fn=None, format_fn=None):
        super(DatasetSequence, self).__init__(CustomDataset(x, y, augment_fn=augment_fn, format_fn=format_fn), batch_size=batch_size, shuffle=True, num_workers=4)