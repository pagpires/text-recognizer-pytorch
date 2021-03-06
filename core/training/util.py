"""Function to train a model."""
from time import time
from typing import Optional

import numpy as np

from torch.utils.data import Dataset
from text_recognizer.models.base import Model
from training.gpu_util_sampler import GPUUtilizationSampler


EARLY_STOPPING = True
GPU_UTIL_SAMPLER = True


def train_model(
        model: Model,
        dataset: Dataset,
        epochs: int,
        batch_size: int,
        gpu_ind: Optional[int] = None,
        use_wandb: bool = False,
        **train_args) -> Model:
    """Train model."""
    callbacks = []

    # TODO: keras specific callbacks
    # if EARLY_STOPPING:
    #     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')
    #     callbacks.append(early_stopping)

    # if GPU_UTIL_SAMPLER and gpu_ind is not None:
    #     gpu_utilization = GPUUtilizationSampler(gpu_ind)
    #     callbacks.append(gpu_utilization)

    # print(model.network)

    t = time()
    _history = model.fit(dataset=dataset, batch_size=batch_size, epochs=epochs, callbacks=callbacks, **train_args)
    print('Training took {:2f} s'.format(time() - t))

    # TODO: util functions
    # if GPU_UTIL_SAMPLER and gpu_ind is not None:
    #     gpu_utilizations = gpu_utilization.samples
    #     print(f'GPU utilization: {round(np.mean(gpu_utilizations), 2)} +- {round(np.std(gpu_utilizations), 2)}')

    return model
