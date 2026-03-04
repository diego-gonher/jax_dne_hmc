####################################################################################################
#
# This contains all the Flax code to train an emulator for the covariance matrix.
#
####################################################################################################

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

from flax import linen as nn
from jax import random as random
import jax
import jax.numpy as jnp
from functools import partial
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
from flax.core import freeze, unfreeze

from sklearn.utils import shuffle

from clu import metrics
from flax.training import train_state, checkpoints  # Useful dataclass to keep train state
from flax import struct  # Flax dataclasses
import optax
from optax import adam, adamw, squared_error

# setting path
import sys
sys.path.append('../')
from dne.diff_emulators.MeanEmulator import MeanEmulator

# use 64 bit precision
# from jax import config
# config.update("jax_enable_x64", True)

# TESTING Disable JIT for now
# from jax.config import config
# config.update('jax_disable_jit', True)
from IPython import embed


####################################################################################################
#                                     HELPER CLASSES                                               #
####################################################################################################

@struct.dataclass
class MetricsCovar(metrics.Collection):
    """A simple extension of clu's Metrics class that only includes the loss metric used for training the model.

    """
    loss: metrics.Average.from_output('loss')
    covar_loss: metrics.Average.from_output('covar_loss')


# holds the model parameters and optimizers, and allows updating it
class TrainStateCovar(train_state.TrainState):
    """A simple extension of Flax's TrainState class to include the metrics. The TrainState class is a dataclass
    that holds the model parameters and optimizers, and allows updating it.

    """
    metrics: MetricsCovar


####################################################################################################
#                                       TRAINER CLASSES                                            #
####################################################################################################

class CovarEmulator(MeanEmulator):

    def __init__(self,
                 model_class: nn.Module,
                 model_hparams: Dict[str, Any],
                 optimizer_class: Any,
                 optimizer_hparams: Dict[str, Any],
                 optimizer_schedule: bool,
                 loss_fn: Callable,
                 X_train: Any,
                 y_train: Any,
                 X_val: Any,
                 y_val: Any,
                 X_scaler_transform: Any = None,
                 y_scaler_inverse_transform: Any = None,
                 num_epochs: int = 25,
                 batch_size: int = 32,
                 seed: int = 42,
                 checkpoint_dir: str = None,
                 **kwargs):
        """
        A basic Trainer module summarizing most common training functionalities.

        Attributes:
          model_class: The class of the model that should be trained.
          model_hparams: A dictionary of all hyperparameters of the model. Is
            used as input to the model when created.
          optimizer_class: Optax optimizer class to use.
          optimizer_hparams: A dictionary of all hyperparameters of the optimizer.
            Used during initialization of the optimizer.
          loss_fn: The loss function to use for training.
          X_train: Training data. Array-like of shape (n_samples, n_features).
          y_train: Training labels. Array-like of shape (n_samples, dim_of_output). Must be preprocessed already.
          X_val: Validation data. Array-like of shape (n_samples, n_features).
          y_val: Validation labels. Array-like of shape (n_samples, dim_of_output). Must be preprocessed already.
          X_scaler_transform: Scaler transformation used for the feature data.
          y_scaler_inverse_transform: Scaler inverse transformation used for the label data.
          seed: Seed to initialize PRNG.
          check_val: Boolean indicating whether to check validation loss.
          checkpoint_dir: Directory to save checkpoints to.
        """
        # initialize from parent class
        super().__init__(model_class=model_class,
                         model_hparams=model_hparams,
                         optimizer_class=optimizer_class,
                         optimizer_hparams=optimizer_hparams,
                         optimizer_schedule=optimizer_schedule,
                         loss_fn=loss_fn,
                         X_train=X_train,
                         y_train=y_train,
                         X_val=X_val,
                         y_val=y_val,
                         X_scaler_transform=X_scaler_transform,
                         y_scaler_inverse_transform=y_scaler_inverse_transform,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         seed=seed,
                         checkpoint_dir=checkpoint_dir,
                         **kwargs)

        # use different metrics for the covariance model (MetricsCovar)
        # Initialize metrics
        self.metrics = MetricsCovar.empty()
        self.metrics_history = {'train_loss': [],
                                'val_loss': [],
                                'train_covar_loss': [],
                                'val_covar_loss': []}  # dictionary with metrics history

        # Initialize train state, this is to overwrite the parent class
        self.init_model_and_train_state(self.exmp_input)

        # get the number of dimensions from the neural network
        self.n_dim = self.model.n_dim

    def init_model_and_train_state(self, exmp_input: Any):
        """
        Initializes the model and tabulates the architecture.

        Args:
          exmp_input: An input to the model with which the shapes are inferred.
        """
        # Prepare PRNG and input
        model_rng = random.PRNGKey(self.seed)
        model_rng, init_rng = random.split(model_rng)
        # Run model initialization
        variables = self.model.init(init_rng, exmp_input)

        print('\nNetwork Architecture:')
        print('Initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(variables)))
        print(self.model.tabulate(jax.random.PRNGKey(0), jnp.ones((1, exmp_input.shape[1]))))

        self.state = TrainStateCovar.create(apply_fn=self.model.apply,
                                            params=variables['params'],
                                            tx=self.optimizer,
                                            metrics=self.metrics)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, batch, debug=False):
        """Train for a single step.

        Args:
          state: TrainState.
          batch: Single batch.
        """

        def inner_loss_fn(params):
            # here, the loss is being calculated on the unscaled flattened Cholesky factors
            predictions = self.y_scaler_inverse_transform(state.apply_fn({'params': params}, batch['X']))
            targets = self.y_scaler_inverse_transform(batch['y'])
            loss = self.loss_fn(predictions, targets=targets).mean()

            return loss

        grad_fn = jax.grad(inner_loss_fn)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)

        return state

    @partial(jax.jit, static_argnums=(0,))
    def compute_metrics(self, *, state, batch, debug=False):
        """Compute the metrics after a single batch.

        Args:
          state: TrainState.
          batch: Single batch.
        """
        # Make predictions and compute the loss
        predictions = self.y_scaler_inverse_transform(state.apply_fn({'params': state.params}, batch['X']))
        targets = self.y_scaler_inverse_transform(batch['y'])
        loss = self.loss_fn(predictions, targets=targets).mean()

        # get the predicted covariance matrices for this batch
        # produce the scaled flattened Cholesky factor, this is simply the output of the network
        network_output_scaled = state.apply_fn({'params': state.params}, batch['X'])
        # convert the output of the network into the original units
        network_output_unscaled = self.y_scaler_inverse_transform(network_output_scaled)
        # get the covariance matrix
        predictions = self.create_covariance_matrix(flat_cholesky_factor_unscaled=network_output_unscaled)

        # reconstruct the target covariance matrices
        targets = self.create_covariance_matrix(
            flat_cholesky_factor_unscaled=self.y_scaler_inverse_transform(batch['y']))

        covar_loss = self.loss_fn(predictions, targets=targets).mean()

        # Compute the metrics
        metric_updates = state.metrics.single_from_model_output(predictions=predictions,
                                                                targets=targets,
                                                                loss=loss,
                                                                covar_loss=covar_loss)

        metrics = state.metrics.merge(metric_updates)
        state = state.replace(metrics=metrics)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def create_covariance_matrix(self, flat_cholesky_factor_unscaled):
        """
        Creates the covariance matrix for the output of the neural network. This is done by first
        creating the covariance matrix in the output space of the neural network and then transforming
        it to the real space.

        Args:
          flat_cholesky_factor_unscaled: Unscaled flat Choelsky factor of the covariance matrix. This is
          the same as the unscaled output of the neural network. Dimensions are given by the following expression
          (batch_size, n_dim + n_dim * (n_dim + 1) / 2). It is assumed that the first n_dim elements are the
          natural log of the diagonal elements of the covariance matrix and the remaining elements are the
          lower triangular elements of the covariance matrix.

        Returns:
          covariance_matrix: The covariance matrix in the real space.
        """
        # get the size of the batch
        batch_size = flat_cholesky_factor_unscaled.shape[0]

        # create an empty array to hold the cholesky factor
        c_factor = jnp.zeros((batch_size, self.n_dim, self.n_dim))

        # indices for diagonal elements and lower triangular elements
        d_idx = jnp.diag_indices(n=self.n_dim)
        l_idx = jnp.tril_indices(n=self.n_dim, k=-1)

        # get the general indices for these
        n_diag = self.n_dim
        n_nondiag = (n_diag * (n_diag - 1)) // 2

        d_idx_general = (jnp.array([[i, ] * n_diag for i in range(batch_size)]).ravel(),)
        d_idx_general += (jnp.array([d_idx[0]] * batch_size).ravel(),)
        d_idx_general += (jnp.array([d_idx[1]] * batch_size).ravel(),)

        l_idx_general = (jnp.array([[i, ] * n_nondiag for i in range(batch_size)]).ravel(),)
        l_idx_general += (jnp.array([l_idx[0]] * batch_size).ravel(),)
        l_idx_general += (jnp.array([l_idx[1]] * batch_size).ravel(),)

        # modify the diagonal and get the lower triangular elements
        diagonal_elements = jnp.exp(flat_cholesky_factor_unscaled[:, :self.n_dim])
        lower_triangular_elements = flat_cholesky_factor_unscaled[:, self.n_dim:]

        # set the diagonal and lower triangular elements
        c_factor = c_factor.at[d_idx_general].set(diagonal_elements.ravel())
        c_factor = c_factor.at[l_idx_general].set(lower_triangular_elements.ravel())

        # get the covariance matrix
        covar_matrix = jnp.matmul(c_factor, c_factor.transpose((0, 2, 1)))

        return jnp.squeeze(covar_matrix)

    def plot_metrics_history(self, save_dir=None, prefix=None):
        """
        Plots the metrics history.
        Args:
          save_dir: Directory to save the plot. If None, the plot is not saved.
          prefix: Prefix to add to the plot name.
        """
        # loss history on the training and validation set calculated for the cholesky factors
        with plt.style.context(['science', 'no-latex']):
            fig, ax = plt.subplots(1, 1, figsize=(7, 3.75))
            ax.plot(self.metrics_history['train_loss'], label='train loss', alpha=0.5)
            ax.plot(self.metrics_history['val_loss'], label='val loss', alpha=0.35, color='red')
            ax.set_title('Loss history')
            ax.set_xlabel('Epoch', fontsize=13)
            ax.set_ylabel('Loss', fontsize=13)
            ax.legend()
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f'{prefix}loss_history.pdf'))
            else:
                plt.show()

        # loss history calculated for the reconstructed covariance matrices
        with plt.style.context(['science', 'no-latex']):
            fig, ax = plt.subplots(1, 1, figsize=(7, 3.75))
            ax.plot(self.metrics_history['train_covar_loss'], label='train loss', alpha=0.5)
            ax.plot(self.metrics_history['val_covar_loss'], label='val loss', alpha=0.35, color='red')
            ax.set_title('Covariance Loss history')
            ax.set_xlabel('Epoch', fontsize=13)
            ax.set_ylabel('Loss', fontsize=13)
            ax.legend()
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f'{prefix}loss_history_on_pred_covar_matrices.pdf'))
            else:
                plt.show()

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, X_unscaled):
        """
        Predicts the output for a given input in 'real' units. That is, both the input X and the output
        y_predicted_unscaled are in the original units of the data.

        Args:
          X_unscaled: Input data in real units (unscaled), dimensions are (n_samples, n_features).

        Returns:
          predicted_covar: The emulated covariance matrix.
        """
        # convert the units of the input data into the scale that the network is expecting them
        X_scaled = self.X_scaler_transform(X_unscaled)
        # produce the scaled flattened Cholesky factor, this is simply the output of the network
        network_output_scaled = self.state.apply_fn({'params': self.state.params}, X_scaled)
        # convert the output of the network into the original units
        network_output_unscaled = self.y_scaler_inverse_transform(network_output_scaled)

        # get the covariance matrix
        predicted_covar = self.create_covariance_matrix(flat_cholesky_factor_unscaled=network_output_unscaled)

        return predicted_covar

    @partial(jax.jit, static_argnums=(0,))
    def predict_single(self, X_unscaled):
        """
        Predicts the output for a single input in 'real' units. That is, both the input X and the output
        y_predicted_unscaled are in the original units of the data.

        Args:
          X_unscaled: Input data in real units (unscaled), dimensions are (n_features,).

        Returns:
          predicted_covar: The emulated covariance matrix.
        """
        # convert the units of the input data into the scale that the network is expecting them
        X_scaled = self.X_scaler_transform(X_unscaled)
        # produce the scaled flattened Cholesky factor, this is simply the output of the network
        flat_cholesky_factor_scaled = self.state.apply_fn({'params': self.state.params}, X_scaled)
        # convert the output of the network into the original units
        flat_cholesky_factor_unscaled = self.y_scaler_inverse_transform(flat_cholesky_factor_scaled)

        # create an empty array to hold the cholesky factor
        c_factor = jnp.zeros((self.n_dim, self.n_dim))

        # indices for diagonal elements and lower triangular elements
        d_idx = jnp.diag_indices(n=self.n_dim)
        l_idx = jnp.tril_indices(n=self.n_dim, k=-1)

        # modify the diagonal and get the lower triangular elements
        diagonal_elements = jnp.exp(flat_cholesky_factor_unscaled[:self.n_dim])
        lower_triangular_elements = flat_cholesky_factor_unscaled[self.n_dim:]

        # set the diagonal and lower triangular elements
        c_factor = c_factor.at[d_idx].set(diagonal_elements)
        c_factor = c_factor.at[l_idx].set(lower_triangular_elements)

        # get the covariance matrix
        predicted_covar_matrix = jnp.matmul(c_factor, c_factor.T)

        return predicted_covar_matrix
        