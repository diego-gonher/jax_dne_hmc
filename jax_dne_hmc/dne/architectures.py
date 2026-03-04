####################################################################################################
#
# This contains the flax neural network architectures.
#
####################################################################################################

from flax import linen as nn
from jax import random as random
import jax
import jax.numpy as jnp
from functools import partial
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union

# use 64 bit precision
# from jax import config
# config.update("jax_enable_x64", True)

# TESTING Disable JIT for now
# from jax.config import config
# config.update('jax_disable_jit', True)
# from IPython import embed


####################################################################################################
#                                     ARCHITECTURES                                                #
####################################################################################################

# A network specifically for the summary statistic emulator
class MeanMLP(nn.Module):
    """A simple MLP implementation using Flax with SiLU (swish) activation function, specifically created
    for the mean of the summary statistic.

    Attributes:
        perceptrons_per_layer (Sequence[int]): List of integers where each element represents the number of perceptrons
            in each hidden layer.
        n_dim (int): The number of dimensions of the summary statistic.

    """
    perceptrons_per_layer: Sequence[int]
    n_dim: int

    @nn.compact
    def __call__(self, inputs):
        # the inputs represent the input layer
        x = inputs

        # use a loop to go through all the hidden layers
        for i, perceptrons in enumerate(self.perceptrons_per_layer):
            x = nn.Dense(perceptrons, name=f'layer_{i}')(x)
            x = nn.swish(x)

        # the end of the MLP, give the appropiate size
        x = nn.Dense(self.n_dim, name=f'summary_statistic')(x)
        x = nn.swish(x)

        return x


# A network specifically for the covariance matrix emulator
class CovarMLP(nn.Module):
    """A simple MLP implementation using Flax with SiLU (swish) activation function, specifically created
    for covariance matrices.

    Attributes:
        perceptrons_per_layer (Sequence[int]): List of integers where each element represents the number of perceptrons
            in each hidden layer.
        n_dim (int): The number of dimensions of the covariance matrix.

    """
    perceptrons_per_layer: Sequence[int]
    n_dim: int

    @nn.compact
    def __call__(self, inputs):
        # the inputs represent the input layer
        x = inputs

        # use a loop to go through all the hidden layers
        for i, perceptrons in enumerate(self.perceptrons_per_layer):
            x = nn.Dense(perceptrons, name=f'layer_{i}')(x)
            x = nn.swish(x)

        # the end of the MLP, give the appropiate size
        x = nn.Dense((self.n_dim * (self.n_dim + 1)) // 2, name=f'input_for_covar')(x)
        x = nn.sigmoid(x)  # the activation function, swish

        return x