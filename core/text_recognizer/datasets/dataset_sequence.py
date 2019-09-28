"""DatasetSequence class."""
import numpy as np
import torch
from torch.utils import data

# def _shuffle(x, y):
#     """Shuffle x and y maintaining their association."""
#     shuffled_indices = np.random.permutation(x.shape[0])
#     return x[shuffled_indices], y[shuffled_indices]

# TODO: this should be a Dataset, then wrap DatasetSequence as data loader
# Create a Dataset that takes EMNIST's self.X, self.y
# https://www.tinymind.com/learn/terms/hdf5#pytorch
# complex version: https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
# Dataset detail https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class CustomDataset(data.Dataset):
    """
    Minimal implementation of https://keras.io/utils/#sequence.
    Allows easy use of fit_generator in training.
    """
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
        # y = torch.from_numpy(np.argmax(self.y[idx, :], axis=-1)).long()
        y = np.argmax(self.y[idx, :], axis=-1).astype(np.uint8)

        if self.augment_fn:
            # read and transformed e2e by transforms
            # do not include ToTensor otherwise y will be rescaled
            # keep io in numpy array
            # BUG this should happen together!!!
            x, y = self.augment_fn(x, y)
            x = np.array(x)
            y = np.array(y)
            
        x = torch.from_numpy(x)
        # TODO decide whether to keep one-hot-encoding or scalar class, or a dataLoader for each dataset
        # TODO this works for lab2-emnist y = torch.from_numpy(self.y[idx, :]).float()

        if x.dtype == torch.uint8:
            # NOTE should tensor.to(float) before division
            x = (x.to(torch.float32) / 255)
        y = torch.from_numpy(y).long()

        # TODO: 
        if self.format_fn:
            x, y = self.format_fn(x, y)

        return x, y

def DatasetSequence(x, y, batch_size=32, augment_fn=None, format_fn=None):
    dl = data.DataLoader(CustomDataset(x, y, augment_fn=augment_fn, format_fn=format_fn), batch_size=batch_size, shuffle=True, num_workers=4)
    return dl
