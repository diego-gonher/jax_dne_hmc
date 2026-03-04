####################################################################################################
#
# Contains loss functions used for both training and evaluating the emulators.
#
####################################################################################################

import numpy as np
import jax.numpy as jnp

# use 64 bit precision
# from jax import config
# config.update("jax_enable_x64", True)

# TESTING Disable JIT for now
# from jax.config import config
# config.update('jax_disable_jit', True)
# from IPython import embed


def rmse(predictions, targets) -> float:
    """Calculates the root mean squared error (RMSE) between the predictions and the targets. This function is designed
    for training, as it returns a scalar value.

    Args:
        predictions (np.ndarray): The predictions, with shape (n_samples, n_features).
        targets (np.ndarray): The targets, with shape (n_samples, n_features).

    Returns:
        float: The RMSE.

    """
    return jnp.sqrt(((predictions - targets) ** 2).mean(axis=1))


def mape(predictions, targets) -> np.ndarray:
    """Calculates the mean absolute percentage error (MAPE) between the predictions and the targets. This function is
    designed for training, as it returns a scalar value.

    Args:
        predictions (np.ndarray): The predictions, with shape (n_samples, n_features).
        targets (np.ndarray): The targets, with shape (n_samples, n_features).

    Returns:
        np.ndarray: The MAPE.

    """
    return jnp.abs((targets - predictions) / targets).mean(axis=1)


def elementwise_mape(predictions, targets) -> np.ndarray:
    """Calculates the absolute percentage error (APE) between the predictions and the targets. This function is
    designed for evaluating the emulator, as it returns a vector of APEs.

    Args:
        predictions (np.ndarray): The predictions, with shape (n_samples, n_features).
        targets (np.ndarray): The targets, with shape (n_samples, n_features).

    Returns:
        np.ndarray: The MAPE.

    """
    return jnp.abs((targets - predictions) / targets)


def relative_rmse(predictions, targets) -> float:
    """Calculates the relative RMSEs between the predictions and the targets. his function is
    designed for training, as it returns a scalar value.

    Args:
        predictions (np.ndarray): The predictions, with shape (n_samples, n_features)..
        targets (np.ndarray): The targets, with shape (n_samples, n_features)..

    Returns:
        float: The relative RMSEs.

    """
    return jnp.sqrt(((predictions - targets) ** 2)/targets**2).mean(axis=1)


def relative_rse(predictions, targets) -> np.ndarray:
    """Calculates the relative root squared error (NO MEAN) between the predictions and the targets. This function is
    designed for evaluating the emulator, as it returns a vector of relative RSEs.

    Args:
        predictions (np.ndarray): The predictions, with shape (n_samples, n_features).
        targets (np.ndarray): The targets, with shape (n_samples, n_features)..

    Returns:
        np.ndarray: The relative RMSEs.

    """
    return jnp.sqrt(((predictions - targets) ** 2)/targets**2)


def mse(predictions, targets) -> float:
    """Calculates the mean squared error (MSE) between the predictions and the targets. This function is designed
    for training, as it returns a scalar value.

    Args:
        predictions (np.ndarray): The predictions, with shape (n_samples, n_features).
        targets (np.ndarray): The targets, with shape (n_samples, n_features).

    Returns:
        float: The MSE.

    """
    return ((predictions - targets) ** 2).mean(axis=1)