####################################################################################################
#
# This is a module intended to create a nice, flexible and simple API for the Flax NNs that we
# will use for our emulators. It also has a class for easy handling of the dataset.
#
####################################################################################################

import os
import json
import matplotlib.pyplot as plt
import scienceplots
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
from flax.training.early_stopping import EarlyStopping
from flax import struct  # Flax dataclasses
import optax
from optax import adam, adamw, squared_error

from jax_dne_hmc.dne.architectures import *
from jax_dne_hmc.dne.losses import *

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

# now we need all the functions required for training
@struct.dataclass
class Metrics(metrics.Collection):
    """A simple extension of clu's Metrics class that only includes the loss metric used for training the model.

    """
    loss: metrics.Average.from_output('loss')


# holds the model parameters and optimizers, and allows updating it
class TrainState(train_state.TrainState):
    """A simple extension of Flax's TrainState class to include the metrics. The TrainState class is a dataclass
    that holds the model parameters and optimizers, and allows updating it.

    """
    metrics: Metrics


####################################################################################################
#                                       TRAINER CLASSES                                            #
####################################################################################################

class MeanEmulator:

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
                 num_epochs: int = 1000,
                 patience: int = 200,
                 batch_size: int = 32,
                 seed: int = 42,
                 checkpoint_dir: str = None,
                 **kwargs):
        """
        A basic Trainer module with the most common training functionalities.

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
          num_epochs: Number of epochs to train.
          patience: Number of epochs with no improvement after which training will be stopped.
          batch_size: Batch size to use during training.
          seed: Seed to initialize PRNG.
          check_val: Boolean indicating whether to check validation loss.
          checkpoint_dir: Directory to save checkpoints to.
        """
        super().__init__()
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_class = optimizer_class
        self.optimizer_hparams = optimizer_hparams
        self.optimizer_schedule = optimizer_schedule
        self.seed = seed
        self.loss_fn = loss_fn
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        if X_scaler_transform is None:
            self.X_scaler_transform = lambda x: x  # do not scale data
        else:
            self.X_scaler_transform = X_scaler_transform
        if y_scaler_inverse_transform is None:
            self.y_scaler_inverse_transform = lambda y: y  # do not scale data
        else:
            self.y_scaler_inverse_transform = y_scaler_inverse_transform
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.exmp_input = self.X_train[:1]
        # Set of hyperparameters to save
        self.config = {
            'model_class': model_class.__name__,
            'model_hparams': model_hparams,
            'optimizer_class': optimizer_class.__name__,
            'optimizer_schedule': optimizer_schedule,
            'loss_fn': loss_fn.__name__,
            'optimizer_hparams': optimizer_hparams,
            'seed': self.seed,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
        }
        self.config.update(kwargs)
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(**self.model_hparams)
        self.state = None
        # Initialize optimizer, check whether to use schedule
        if self.optimizer_schedule:
            hparams_for_schedule = self.optimizer_hparams.copy()
            lr_ = hparams_for_schedule.pop('learning_rate')
            warmup_ = hparams_for_schedule.pop('warmup', 0)
            num_steps_per_epoch = X_train.shape[0] // batch_size
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr_,
                warmup_steps=warmup_,
                decay_steps=int(num_epochs * num_steps_per_epoch),
                end_value=0.01 * lr_
            )
            self.optimizer = self.optimizer_class(lr_schedule, **hparams_for_schedule)
        else:
            self.optimizer = self.optimizer_class(**self.optimizer_hparams)
        # Initialize metrics
        self.metrics = Metrics.empty()
        self.metrics_history = {'train_loss': [], 'val_loss': []}  # dictionary with metrics history
        # Initialize train state
        self.init_model_and_train_state(self.exmp_input)
        # Save hyperparameters
        with open(os.path.join(self.checkpoint_dir, 'hparams.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        # jit the loss function
        self.loss_fn = jax.jit(self.loss_fn)

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

        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=variables['params'],
                                       tx=self.optimizer,
                                       metrics=self.metrics)

    def create_batches(self, X, y, rstate, batch_size):
        """
        Creates batches and returns them as dictionaries for easier use.

        Args:
          X: Array of predictors.
          y: Array of labels.
          rstate: Random state for shuffling.
          batch_size: Size of the batches.
        """
        # first we shuffle the data
        X, y = shuffle(X, y, random_state=rstate)
        # calculate the number of batches, TODO: make this more flexible
        n_batches = X.shape[0] // batch_size

        batches = []

        for i in np.arange(n_batches):
            single_batch = {
                'X': X[i * batch_size:(i + 1) * batch_size, :].reshape((batch_size, X.shape[1])),
                'y': y[i * batch_size:(i + 1) * batch_size]}
            batches.append(single_batch)

        return batches

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, batch):
        """Train for a single step.

        Args:
          state: TrainState.
          batch: Single batch.
        """

        def inner_loss_fn(params):
            predictions = self.y_scaler_inverse_transform(state.apply_fn({'params': params}, batch['X']))
            targets = self.y_scaler_inverse_transform(batch['y'])
            loss = self.loss_fn(predictions, targets=targets).mean()
            return loss

        grad_fn = jax.grad(inner_loss_fn)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    @partial(jax.jit, static_argnums=(0,))
    def compute_metrics(self, *, state, batch):
        """Compute the metrics after a single batch.

        Args:
          state: TrainState.
          batch: Single batch.
        """
        # Make predictions and compute the loss
        predictions = self.y_scaler_inverse_transform(state.apply_fn({'params': state.params}, batch['X']))
        targets = self.y_scaler_inverse_transform(batch['y'])
        loss = self.loss_fn(predictions, targets=targets).mean()

        # Compute the metrics
        metric_updates = state.metrics.single_from_model_output(predictions=predictions, targets=batch['y'], loss=loss)
        metrics = state.metrics.merge(metric_updates)
        state = state.replace(metrics=metrics)

        return state

    def train(self):
        """
        Train the model.
        """

        print(f'\n########## Training for {self.num_epochs} epochs ##########\n')

        best_eval_metrics = None
        new_metrics = None
        patience_counter = 0

        for e in np.arange(self.num_epochs):
            # create randomly shuffled batches
            all_batches = self.create_batches(self.X_train, self.y_train, rstate=e, batch_size=self.batch_size)

            # go through each batch
            for batch in all_batches:
                # Run optimization steps over training batches and compute batch metrics
                self.state = self.train_step(state=self.state,
                                             batch=batch)  # get updated train state (with updated parameters)

                self.state = self.compute_metrics(state=self.state,
                                                  batch=batch)  # aggregate batch metrics

            # after going through all the batches, an epoch has ended
            for metric, value in self.state.metrics.compute().items():  # compute metrics
                self.metrics_history[f'train_{metric}'].append(value)  # record metrics

            self.state = self.state.replace(metrics=self.state.metrics.empty())  # reset the_metrics for next epoch

            # string for updating the user
            string_epoch = f"train epoch: {e + 1}"
            string_test_loss = f"train_loss: {self.metrics_history['train_loss'][-1]}"
            string_update = string_epoch + ",   " + string_test_loss

            # calculate the loss in the validation set if it is requested
            if self.X_val is not None and self.y_val is not None:
                # create randomly shuffled batches
                validation_batch = {'X': self.X_val, 'y': self.y_val}

                # compute metrics on the validation set after each training epoch
                test_state = self.state
                test_state = self.compute_metrics(state=test_state, batch=validation_batch)

                for metric, value in test_state.metrics.compute().items():
                    # embed(header='check')
                    self.metrics_history[f'val_{metric}'].append(value)

                string_update += f",   val_loss: {self.metrics_history['val_loss'][-1]}"

            # save the model if it is the best one so far
            # first, we check if we are using the validation set or not
            if self.X_val is not None and self.y_val is not None:
                new_metrics = self.metrics_history['val_loss'][-1]
            else:
                new_metrics = self.metrics_history['train_loss'][-1]

            # save the model if it is the best one so far
            if best_eval_metrics is None or new_metrics < best_eval_metrics:
                string_update += "   (saving model)"
                best_eval_metrics = new_metrics
                patience_counter = 0
                self.save_checkpoint(step=e + 1)
            else:
                patience_counter += 1

            # string_update += f",   pat_count: {self.early_stop.patience_count}, stop: {self.early_stop.should_stop}"

            # stop the training if the patience counter has reached the self.patience limit
            if patience_counter >= self.patience:
                print(f'Met early stopping criteria, breaking at epoch {e}')
                break

            # print the update
            print(string_update)

        # return the best evaluation metrics, this is for hyperparameter optimization
        print(f'\n########## Training finished ##########\n')
        print(f'Best evaluation metrics: {best_eval_metrics}\n')

        return best_eval_metrics

    def bind_model(self):
        """
        Returns a model with parameters bound to it. Enables an easier inference
        access.

        Returns:
          The model with parameters bound to it.
        """
        params = {'params': self.state.params}
        return self.model.bind(params)

    def save_checkpoint(self, step: int):
        """
        Saves current training state at certain training iteration. Only the model
        parameters are saved to reduce memory footprint.
        To support the training to be continued from a checkpoint, this method can be
        extended to include the optimizer state as well.

        Args:
          step: Index of the step to save the model at, e.g. epoch.
        """
        checkpoints.save_checkpoint(ckpt_dir=self.checkpoint_dir,
                                    target={'params': self.state.params},
                                    step=step,
                                    overwrite=True,
                                    keep=1)

    def load_checkpoint(self):
        """
        Loads model parameters from a saved checkpoint.
        """
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.checkpoint_dir, target=None)

        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict['params'],
                                       tx=self.optimizer,
                                       metrics=self.metrics)

    @classmethod
    def load_model(cls, checkpoint_dir, X_train, y_train, X_val, y_val,
                   X_scaler_transform, y_scaler_inverse_transform):
        """
        Loads model parameters from the logging directory.

        Args:
          checkpoint_dir: Directory where the model is saved.
          X_train: Training predictors.
          y_train: Training labels.
          X_val: Validation predictors.
          y_val: Validation labels.
          X_scaler_transform: Scaler transformation used for the feature data.
          y_scaler_inverse_transform: Scaler inverse transformation used for the label data.

        Returns:
          A Trainer object with model loaded from the checkpoint folder.
        """

        hparams_file = os.path.join(checkpoint_dir, 'hparams.json')
        with open(hparams_file, 'r') as f:
            hparams = json.load(f)

        model_class = eval(hparams.pop('model_class'))
        optimizer_class = eval(hparams.pop('optimizer_class'))
        loss_fn = eval(hparams.pop('loss_fn'))

        trainer = cls(model_class=model_class,
                      optimizer_class=optimizer_class,
                      loss_fn=loss_fn,
                      checkpoint_dir=checkpoint_dir,
                      X_train=X_train,
                      y_train=y_train,
                      X_val=X_val,
                      y_val=y_val,
                      X_scaler_transform=X_scaler_transform,
                      y_scaler_inverse_transform=y_scaler_inverse_transform,
                      **hparams)

        trainer.load_checkpoint()

        return trainer

    @classmethod
    def load_pretrained_model(cls, checkpoint_dir, X_train, y_train, X_val, y_val,
                              X_scaler_transform, y_scaler_inverse_transform,
                              optimizer_schedule, num_epochs, batch_size, optimizer_hparams):
        """
        Loads model parameters from the logging directory.

        Args:
          checkpoint_dir: Directory where the model is saved.
          X_train: Training predictors.
          y_train: Training labels.
          X_val: Validation predictors.
          y_val: Validation labels.
          X_scaler_transform: Scaler transformation used for the feature data.
          y_scaler_inverse_transform: Scaler inverse transformation used for the label data.
          optimizer_schedule: Learning rate schedule for the optimizer. Boolean.
          num_epochs: Number of epochs for fine-tuning.
          batch_size: Batch size for fine-tuning.
          optimizer_hparams: Hyperparameters for the optimizer.

        Returns:
          A Trainer object with model loaded from the checkpoint folder.
        """
        # load the pre-trained model
        pre_trained_model = cls.load_model(checkpoint_dir,
                                           X_train,
                                           y_train,
                                           X_val,
                                           y_val,
                                           X_scaler_transform,
                                           y_scaler_inverse_transform)

        # set the new training parameters
        pre_trained_model.optimizer_schedule = optimizer_schedule
        pre_trained_model.num_epochs = num_epochs
        pre_trained_model.batch_size = batch_size
        pre_trained_model.optimizer_hparams = optimizer_hparams

        # re-initialize the optimizer
        # Initialize optimizer, check whether to use schedule
        if pre_trained_model.optimizer_schedule:
            hparams_for_schedule = pre_trained_model.optimizer_hparams.copy()
            lr_ = hparams_for_schedule.pop('learning_rate')
            warmup_ = hparams_for_schedule.pop('warmup', 0)
            num_steps_per_epoch = X_train.shape[0] // batch_size
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr_,
                warmup_steps=warmup_,
                decay_steps=int(num_epochs * num_steps_per_epoch),
                end_value=0.01 * lr_
            )
            pre_trained_model.optimizer = pre_trained_model.optimizer_class(lr_schedule, **hparams_for_schedule)
        else:
            pre_trained_model.optimizer = pre_trained_model.optimizer_class(**pre_trained_model.optimizer_hparams)

        return pre_trained_model

    def plot_metrics_history(self, save_dir=None, prefix=None):
        """
        Plots the metrics history.
        Args:
          save_dir: Directory to save the plot. If None, the plot is not saved.
          prefix: Prefix to add to the plot name.
        """
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

    def predict(self, X_unscaled):
        """
        Predicts the output for a given input in 'real' units. That is, both the input X and the output
        y_predicted_unscaled are in the original units of the data.

        Args:
          X_unscaled: Input data in real units (unscaled).

        Returns:
          y_predicted_unscaled: The output of the model also in real units.
        """
        X_scaled = self.X_scaler_transform(X_unscaled)
        y_predicted = self.state.apply_fn({'params': self.state.params}, X_scaled)
        y_predicted_unscaled = self.y_scaler_inverse_transform(y_predicted)

        return y_predicted_unscaled

