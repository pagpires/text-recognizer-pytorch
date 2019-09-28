"""Define LineDetectorModel class."""
from typing import Callable, Dict, Tuple

import numpy as np

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as tvf

from text_recognizer.datasets.dataset_sequence import DatasetSequence
from text_recognizer.datasets.iam_paragraphs_dataset import IamParagraphsDataset
from text_recognizer.models.base import Model
from text_recognizer.networks import fcn


# _DATA_AUGMENTATION_PARAMS = {
#     'width_shift_range': 0.06,
#     'height_shift_range': 0.1,
#     'horizontal_flip': True,
#     'zoom_range': 0.1,
#     'fill_mode': 'constant',
#     'cval': 0,
#     'shear_range': 3,
# }

# transformation should happen before/after ToTensor?
param_dict = {
    'degrees':(0, 0), 'translate': (0.06, 0.1), 'scale_ranges':(0.9, 1.1), 'shears': (-1.5, 1.5)
}
def pair_transform(image, mask):
    image, mask = transforms.ToPILImage()(image), transforms.ToPILImage()(mask)
    ret = transforms.RandomAffine.get_params(img_size=image.size, **param_dict)
    image = tvf.affine(image, *ret, resample=0, fillcolor=0)
    mask = tvf.affine(mask, *ret, resample=0, fillcolor=0)
    if np.random.random() > 0.5:
        image, mask = tvf.hflip(image), tvf.hflip(mask)
    return image, mask

trsfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=0, translate=(0.06, 0.1), scale=(0.9, 1.1), shear=1.5, fillcolor=0), 
    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor()
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LineDetectorModel(Model):
    """Model to detect lines of text in an image."""
    def __init__(self,
                 dataset_cls: type = IamParagraphsDataset,
                 network_fn: Callable = fcn,
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

        # self.data_augmentor = ImageDataGenerator(**_DATA_AUGMENTATION_PARAMS)
        # self.batch_augment_fn = self.augment_batch
        self.batch_augment_fn = pair_transform

    def loss(self):  # pylint: disable=no-self-use
        return nn.NLLLoss

    def metrics(self):  # pylint: disable=no-self-use
        return None

    # def augment_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """Performs different random transformations on the whole batch of x, y samples."""
    #     x_augment, y_augment = zip(*[self._augment_sample(x, y) for x, y in zip(x_batch, y_batch)])
    #     return np.stack(x_augment, axis=0), np.stack(y_augment, axis=0)

    # def _augment_sample(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Perform the same random image transformation on both x and y.
    #     x is a 2d image of shape self.image_shape, but self.data_augmentor needs the channel image too.
    #     """
    #     x_3d = np.expand_dims(x, axis=-1)
    #     transform_parameters = self.data_augmentor.get_random_transform(x_3d.shape)
    #     x_augment = self.data_augmentor.apply_transform(x_3d, transform_parameters)
    #     y_augment = self.data_augmentor.apply_transform(y, transform_parameters)
    #     return np.squeeze(x_augment, axis=-1), y_augment

    def predict_on_image(self, x: np.ndarray) -> np.ndarray:
        """Returns the network predictions on x."""
        # return self.network.predict(np.expand_dims(x, axis=0))[0]
        # read np array, convert from hw to nhw
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            pred = self.network(torch.from_numpy(x).unsqueeze(0).to(device))[0].cpu()
        if was_training:
            self.network.train()

        return pred

    # def evaluate(self,
    #              x: np.ndarray,
    #              y: np.ndarray,
    #              batch_size: int = 32,
    #              verbose: bool = False) -> float:  # pylint: disable=unused-argument
    #     """Evaluates the network on x, y on returns the loss."""
    #     return self.network.evaluate(x, y, batch_size=batch_size)
    
    def evaluate(self, x, y, batch_size=16, verbose=False):
        # NOTE no transformation here
        val_dl = DatasetSequence(x, y, batch_size=batch_size)
        was_training = self.network.training
        self.network.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in val_dl:
                batch_inputs, batch_labels = batch
                batch_inputs = batch_inputs.to(device)
                # batch_labels = torch.squeeze(batch_labels, dim=1) # no need to move label to GPU

                batch_preds = self.network(batch_inputs)
                preds.append(batch_preds.cpu())
                labels.append(batch_labels)

        # TODO optimize the conversion btw cpu and cuda
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        
        if was_training:
            self.network.train()

        n, h, w = labels.shape
        # preds: (batch, num_class, h, w); labels: (batch, h, w)
        cors = np.argmax(preds, axis=1) == labels
        mcors = np.sum(cors, axis=(1,2)) / (h*w)
        return np.mean(mcors)
        # return np.mean(np.sum(np.argmax(preds, axis=1) == labels, axis=(1,2)) / (h*w))