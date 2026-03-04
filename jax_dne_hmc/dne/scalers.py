####################################################################################################
#
# This is a file that contains the differentiable scalers used for data processing.
#
####################################################################################################
import jax
import jax.numpy as jnp
from functools import partial

# use 64 bit precision
# from jax import config
# config.update("jax_enable_x64", True)

# TESTING Disable JIT for now
# from jax.config import config
# config.update('jax_disable_jit', True)
# from IPython import embed


####################################################################################################
#                                       SCALER CLASSES                                             #
####################################################################################################

# a differentiable version of the minmax scaler
class DiffMinMaxScaler:
    """
    A differentiable minmax scaler for use in JAX.

    ...

    Attributes
    ----------
    min : float
        the minimum value of the dataset
    max : float
        the maximum value of the dataset

    Methods
    -------
    fit(dataset):
        Obtains the parameters used in the min max scaling.
    transform(dataset):
        Returns the dataset but transformed such that each feature is min max scaled.
    inverse_transform(dataset):
        Performs the inverse transformation such that you can recover the unscaled dataset from a scaled version of it.
    """

    def __init__(self):
        """
        Initializes the min and max values to None.
        """
        # initialize the min and max values
        self.min = None
        self.max = None

    def fit(self, dataset):
        """
        Obtains the parameters used in the min max scaling.

        Args:
          dataset: dataset that is going to be transformed.
        """
        # obtain the min and max per feature
        self.min = dataset.min(axis=0)
        self.max = dataset.max(axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def transform(self, dataset):
        """
        Returns the dataset but transformed such that each
        feature is min max scaled.

        Args:
          dataset: dataset that is going to be transformed.
        """
        return (dataset - self.min) / (self.max - self.min)

    @partial(jax.jit, static_argnums=(0,))
    def inverse_transform(self, dataset):
        """
        Performs the inverse transformation such that you
        can recover the unscaled dataset from a scaled
        version of it.

        Args:
          dataset: dataset that is going to be transformed.
        """
        return (dataset * (self.max - self.min)) + self.min


class DiffStandardScaler:
    """
    A differentiable minmax scaler for use in JAX.

    ...

    Attributes
    ----------
    mean : float
        the mean value of the dataset
    std : float
        the standard deviation value of the dataset

    Methods
    -------
    fit(dataset):
        Obtains the parameters used in the min max scaling.
    transform(dataset):
        Returns the dataset but transformed such that each feature is min max scaled.
    inverse_transform(dataset):
        Performs the inverse transformation such that you can recover the unscaled dataset from a scaled version of it.
    """

    def __init__(self):
        """
        Initializes the min and max values to None.
        """
        # initialize the mean and std values
        self.mean = None
        self.std = None

    def fit(self, dataset):
        """
        Obtains the parameters used in the min max scaling.

        Args:
          dataset: dataset that is going to be transformed.
        """
        # obtain the min and max per feature
        self.mean = dataset.mean(axis=0).reshape(1, -1)
        self.std = dataset.std(axis=0).reshape(1, -1)

    @partial(jax.jit, static_argnums=(0,))
    def transform(self, dataset):
        """
        Returns the dataset but transformed such that each
        feature is min max scaled.

        Args:
          dataset: dataset that is going to be transformed.
        """
        return (dataset - self.mean) / self.std

    @partial(jax.jit, static_argnums=(0,))
    def inverse_transform(self, dataset):
        """
        Performs the inverse transformation such that you
        can recover the unscaled dataset from a scaled
        version of it.

        Args:
          dataset: dataset that is going to be transformed.
        """
        return (dataset * self.std) + self.mean