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

# transformation should happen before ToTensor
param_dict = {
    "degrees": (0, 0),
    "translate": (0.06, 0.1),
    "scale_ranges": (0.9, 1.1),
    "shears": (-1.5, 1.5),
}


def pair_transform(image, mask):
    image, mask = transforms.ToPILImage()(image), transforms.ToPILImage()(mask)
    ret = transforms.RandomAffine.get_params(img_size=image.size, **param_dict)
    image = tvf.affine(image, *ret, resample=0, fillcolor=0)
    mask = tvf.affine(mask, *ret, resample=0, fillcolor=0)
    if np.random.random() > 0.5:
        image, mask = tvf.hflip(image), tvf.hflip(mask)
    return image, mask


trsfm = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomAffine(
            degrees=0, translate=(0.06, 0.1), scale=(0.9, 1.1), shear=1.5, fillcolor=0
        ),
        transforms.RandomHorizontalFlip(),
    ]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LineDetectorModel(Model):
    """Model to detect lines of text in an image."""

    def __init__(
        self,
        dataset_cls: type = IamParagraphsDataset,
        network_fn: Callable = fcn,
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

        self.batch_augment_fn = pair_transform

    def loss(self):  # pylint: disable=no-self-use
        return nn.NLLLoss

    def metrics(self):  # pylint: disable=no-self-use
        return None

    def predict_on_image(self, x: np.ndarray) -> np.ndarray:
        """Returns the network predictions on x."""
        # return self.network.predict(np.expand_dims(x, axis=0))[0]
        # read np array, convert from hw to nhw
        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            pred = (
                self.network(torch.from_numpy(x).unsqueeze(0).to(device))[0]
                .cpu()
                .numpy()
            )
        if was_training:
            self.network.train()

        return pred

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
                batch_inputs = batch_inputs.to(device)  # no need to move label to GPU

                batch_preds = self.network(batch_inputs)
                preds.append(batch_preds.cpu())
                labels.append(batch_labels)

        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()

        if was_training:
            self.network.train()

        n, h, w = labels.shape
        # preds: (batch, num_class, h, w); labels: (batch, h, w)
        corrects = np.argmax(preds, axis=1) == labels
        mean_corrects = np.sum(corrects, axis=(1, 2)) / (h * w)
        return np.mean(mean_corrects)
