"""Model class, to be extended by specific types of models."""
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

from text_recognizer.datasets.dataset_sequence import Dataset

DIRNAME = Path(__file__).parents[1].resolve() / 'weights'


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""
    def __init__(self, dataset_cls: type, network_fn: Callable, dataset_args: Dict = None, network_args: Dict = None):
        self.name = f'{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}'

        if dataset_args is None:
            dataset_args = {}
        self.data = dataset_cls(**dataset_args)

        if network_args is None:
            network_args = {}
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)

        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

    @property
    def image_shape(self):
        return self.data.input_shape

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f'{self.name}_weights.h5')

    # TODO: rewrite fit
    # dataset contains both train and test data
    def fit(self, dataset, batch_size: int = 32, epochs: int = 10, augment_val: bool = True, callbacks: list = None):
        if callbacks is None:
            callbacks = []

        # TODO: set up optimizer, loss
        # optimizer_type = self.optimizer()
        # optimizer = optimizer_type(self.network.parameters())
        # loss_fn_type = self.loss()
        # loss_fn = loss_fn_type()
        optimizer = optim.Adam(self.network.parameters(), lr=3e-4)
        loss_fn = nn.BCEWithLogitsLoss()

        self.network.train()

        # TODO: 2 step, assign a dataset, then convert to DataLoader()
        # modify dataset to make it contain train and test

        train_data = Dataset(
            dataset.x_train,
            dataset.y_train,
            augment_fn=self.batch_augment_fn,
            format_fn=self.batch_format_fn
        )
        train_sequence = DataLoader(
            train_data,
            batch_size,
            shuffle=True,
            num_workers=4
        )


        # TODO: prepare test dataset
        # test_sequence = DatasetSequence(
        #     dataset.x_test,
        #     dataset.y_test,
        #     batch_size,
        #     augment_fn=self.batch_augment_fn if augment_val else None,
        #     format_fn=self.batch_format_fn
        # )

        # import pdb; pdb.set_trace()

        interval = 10
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, batch in enumerate(train_sequence, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = batch
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

            if epoch % interval == (interval-1):    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss / interval))
                running_loss = 0.0

            # TODO: add validation
        
        print('Finished Training')

        # TODO: for loop to run training
        # self.network.fit_generator(
        #     generator=train_sequence,
        #     epochs=epochs,
        #     callbacks=callbacks,
        #     validation_data=test_sequence,
        #     use_multiprocessing=True,
        #     workers=2,
        #     shuffle=True
        # )
        
    # TODO
    def evaluate(self, x, y, batch_size=16, verbose=False):  # pylint: disable=unused-argument
        val_dl = DataLoader(Dataset(x, y), batch_size=batch_size)  # Use a small batch size to use less memory
        was_training = self.network.training
        self.network.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in val_dl:
                batch_inputs, batch_labels = batch
                batch_labels = torch.squeeze(batch_labels, dim=1)
                batch_preds = self.network(batch_inputs)
                preds.append(batch_preds)
                labels.append(batch_labels)
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()
        
        if was_training:
            self.network.train()
        return np.mean(np.argmax(preds, -1) == np.argmax(labels, -1))

    def loss(self):  # pylint: disable=no-self-use
        return nn.BCELoss

    def optimizer(self):  # pylint: disable=no-self-use
        return RMSprop

    def metrics(self):  # pylint: disable=no-self-use
        return ['accuracy']

    def load_weights(self):
        params = torch.load(self.weights_filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(params['state_dict'])

    def save_weights(self):
        print(f'save model parameters to [{self.weights_filename}]')
        params = {
            'state_dict': self.network.state_dict()
        }
        torch.save(params, self.weights_filename)

