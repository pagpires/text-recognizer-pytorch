"""Define LineModel class."""
from typing import Callable, Dict, Tuple

import editdistance
import numpy as np

import torch

from text_recognizer.datasets.emnist_lines_dataset import EmnistLinesDataset
from text_recognizer.datasets.dataset_sequence import DatasetSequence
from text_recognizer.models.base import Model
from text_recognizer.networks import line_cnn_all_conv
from text_recognizer.util import to_categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LineModel(Model):
    """Model for predicting a string from an image of a handwritten line of text."""

    def __init__(
        self,
        dataset_cls: type = EmnistLinesDataset,
        network_fn: Callable = line_cnn_all_conv,
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        """Define the default dataset and network values for this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def loss(self):
        return torch.nn.NLLLoss

    def evaluate(self, x, y, batch_size=16, verbose=True):
        # x: (n, h, w); y: (n, output_length, num_classes)
        num_data, output_length, num_classes = y.shape
        loss_fn = self.loss()()
        sequence = DatasetSequence(x, y, batch_size=batch_size)

        running_loss = 0
        with torch.no_grad():
            was_training = self.network.training
            self.network.eval()
            preds_raw = []
            labels_raw = []
            for i, batch in enumerate(sequence):
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                batch_pred = self.network(batch_x)
                loss = loss_fn(batch_pred, batch_y)
                running_loss += loss.item()
                preds_raw.append(batch_pred.cpu())
                labels_raw.append(batch_y.cpu())

            preds_raw = torch.cat(preds_raw).numpy()
            # transform labels from scalar to original one-hot-encoding shape
            # labels_raw = to_categorical(torch.cat(labels_raw).numpy(), num_classes)
            print(f"Evaluation loss: {running_loss/(i+1)}")
            if was_training:
                self.network.train()

        # trues.shape: (batch, output_length, num_classes)
        # preds_raw.shape = (batch, num_classes, output_length)
        trues = torch.cat(labels_raw).numpy()
        preds = np.argmax(preds_raw, 1)
        pred_strings = [
            "".join(self.data.mapping.get(label, "") for label in pred).strip(" |_")
            for pred in preds
        ]
        true_strings = [
            "".join(self.data.mapping.get(label, "") for label in true).strip(" |_")
            for true in trues
        ]
        char_accuracies = [
            1 - editdistance.eval(true_string, pred_string) / len(true_string)
            for pred_string, true_string in zip(pred_strings, true_strings)
        ]
        if verbose:
            sorted_ind = np.argsort(char_accuracies)
            print("\nLeast accurate predictions:")
            for ind in sorted_ind[:5]:
                print(f"True: {true_strings[ind]}")
                print(f"Pred: {pred_strings[ind]}")
            print("\nMost accurate predictions:")
            for ind in sorted_ind[-5:]:
                print(f"True: {true_strings[ind]}")
                print(f"Pred: {pred_strings[ind]}")
            print("\nRandom predictions:")
            random_ind = np.random.randint(0, len(char_accuracies), 5)
            for ind in random_ind:  # pylint: disable=not-an-iterable
                print(f"True: {true_strings[ind]}")
                print(f"Pred: {pred_strings[ind]}")
        mean_accuracy = np.mean(char_accuracies)

        return mean_accuracy

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)
        image = torch.from_numpy(np.expand_dims(image, 0)).to(device)
        with torch.no_grad():
            was_training = self.network.training
            self.network.eval()
            pred_raw = self.network(image).squeeze(dim=0).cpu().numpy()
            if was_training:
                self.network.train()
        # pred_raw: (num_classes, output_length)
        pred = "".join(
            self.data.mapping[label] for label in np.argmax(pred_raw, axis=0).flatten()
        ).strip(" |_")
        conf = np.exp(
            np.min(np.max(pred_raw, axis=0))
        )  # The least confident of the predictions.
        return pred, conf
