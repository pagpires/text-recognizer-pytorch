"""LeNet network."""
from typing import Tuple

from torch import nn

# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D
# from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
    """Return LeNet Keras model."""
    num_classes = output_shape[0]

    # Your code below (Lab 2)

    # Your code above (Lab 2)

    return model
