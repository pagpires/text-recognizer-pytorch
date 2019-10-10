"""Define CharacterModel class."""
from typing import Callable, Dict, Tuple

import numpy as np
import torch

from text_recognizer.models.base import Model
from text_recognizer.datasets.emnist_dataset import EmnistDataset
from text_recognizer.networks.mlp import mlp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CharacterModel(Model):
    def __init__(self,
                 dataset_cls: type = EmnistDataset,
                 network_fn: Callable = mlp,
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        # NOTE: integer to character mapping dictionary is self.data.mapping[integer]
        with torch.no_grad():
            was_training = self.network.training
            self.network.eval()
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            pred_raw = self.network(image).cpu().numpy()[0]
            ind = np.argmax(pred_raw)
            confidence_of_prediction = np.exp(pred_raw[ind])
            predicted_character = self.data.mapping[ind]
        if was_training: self.network.train()
        return predicted_character, confidence_of_prediction
