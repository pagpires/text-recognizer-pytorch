"""Model class, to be extended by specific types of models."""
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch import nn, optim
import numpy as np

from text_recognizer.datasets.dataset_sequence import DatasetSequence

DIRNAME = Path(__file__).parents[1].resolve() / "weights"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(
        self,
        dataset_cls: type,
        network_fn: Callable,
        dataset_args: Dict = None,
        network_args: Dict = None,
    ):
        self.name = (
            f"{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}"
        )

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(
            self.data.input_shape, self.data.output_shape, **network_args
        )

        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

    @property
    def image_shape(self):
        return self.data.input_shape

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f"{self.name}_weights.h5")

    def fit(
        self,
        dataset,
        batch_size: int = 32,
        epochs: int = 10,
        augment_val: bool = True,
        callbacks: list = None,
        **train_args,
    ):
        if callbacks is None:
            callbacks = []
        if train_args.get("pretrained", False):
            self.load_weights()
            print("loaded pretrained network")

        train_sequence = DatasetSequence(
            dataset.x_train,
            dataset.y_train,
            batch_size=batch_size,
            augment_fn=self.batch_augment_fn,
            format_fn=self.batch_format_fn,
        )

        print(f"Total #training: {len(train_sequence.dataset)}")
        print(
            f"Total #params: {sum([param.nelement() for param in self.network.parameters()])}"
        )

        self.network.to(device)
        self.network.train()

        optimizer_class = self.optimizer()
        optimizer = optimizer_class(self.network.parameters(), lr=3e-4)  # magic Adam lr
        loss_fn_class = self.loss()
        loss_fn = loss_fn_class()

        validation_interval = 1
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, batch in enumerate(train_sequence, 0):

                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = torch.squeeze(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.network(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            print(f"[{epoch+1}, {i+1}] loss: {running_loss/(i+1):.5f}")

            if epoch % validation_interval == (validation_interval - 1):
                score = self.evaluate(dataset.x_test, dataset.y_test)
                print(f"Validation score: {score:.4f}")

        print("Finished Training")

    def evaluate(self, x, y, batch_size=64, verbose=False):
        val_dl = DatasetSequence(x, y, batch_size=batch_size)
        was_training = self.network.training
        self.network.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in val_dl:
                batch_inputs, batch_labels = batch
                batch_inputs = batch_inputs.to(device)
                batch_labels = torch.squeeze(
                    batch_labels, dim=1
                )  # no need to move label to GPU

                batch_preds = self.network(batch_inputs)
                preds.append(batch_preds.cpu())
                labels.append(batch_labels)

        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()  # (B,)

        if was_training:
            self.network.train()
        print(f"Evaluated: preds: {preds.shape}, labels: {labels.shape}")
        return np.mean(np.argmax(preds, -1) == labels)

    def loss(self):
        return nn.NLLLoss

    def optimizer(self):
        return optim.Adam

    def metrics(self):
        return ["accuracy"]

    def load_weights(self):
        params = torch.load(
            self.weights_filename, map_location=lambda storage, loc: storage
        )
        self.network.load_state_dict(params["state_dict"])
        self.network.to(device)

    def save_weights(self):
        print(f"save model parameters to [{self.weights_filename}]")
        params = {"state_dict": self.network.state_dict()}
        torch.save(params, self.weights_filename)
