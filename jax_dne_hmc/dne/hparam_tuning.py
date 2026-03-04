import os
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np

import logging
import sys

import jax.numpy as jnp
import optax
import optuna
from optuna.samplers import TPESampler

from jax_dne_hmc.dne.architectures import MeanMLP
from jax_dne_hmc.dne.mean_emulator import MeanEmulator
from jax_dne_hmc.dne.architectures import CovarMLP
from jax_dne_hmc.dne.covariance_emulator import CovarEmulator
from dne.losses import mape, rmse, relative_rmse, mse

from IPython import embed


####################################################################################################
# a function to do hyperparameter tuning for the mean emulator
####################################################################################################

class HParamTunerMean:

    def __init__(self, hparam_tuning_dict, loss_fn, batch_size, ntrials, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                 scaler_X_transform, scaler_y_inverse_transform, checkpoint_directory, hparam_results_directory,
                 best_model_directory, model_perfromance_directory, emulator_seed=42, study_seed=10):
        """
        Does hyperparameter tuning for the mean emulator, saves the best model, and returns it.

        Args:
          hparam_tuning_dict: A dictionary containing the hyperparameters to be tuned.
          loss_fn: The loss function used to train the emulator.
          batch_size: The batch size used during training.
          ntrials: The number of trials to run.
          X_train_scaled: The input data scaled.
          y_train_scaled: The output data scaled.
          X_val_scaled: The input data scaled for the validation set.
          y_val_scaled: The output data scaled for the validation set.
          scaler_X_transform: The scaler used to scale the input data.
          scaler_y_inverse_transform: The scaler used to scale the output data.
          checkpoint_directory: The directory where the checkpoints are saved.
          hparam_results_directory: The directory where the hparam tuning results are saved.
          best_model_directory: The directory where the best model is saved.
          model_perfromance_directory: The directory where the model performance is saved.
          emulator_seed: The seed used for the emulator.
          study_seed: The seed used for the hyper-parameter study.

        Returns:
          the trained best emulator.
        """
        # save the input arguments
        self.hparam_tuning_dict = hparam_tuning_dict
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.ntrials = ntrials
        self.X_train_scaled = X_train_scaled
        self.y_train_scaled = y_train_scaled
        self.X_val_scaled = X_val_scaled
        self.y_val_scaled = y_val_scaled
        self.scaler_X_transform = scaler_X_transform
        self.scaler_y_inverse_transform = scaler_y_inverse_transform
        self.checkpoint_directory = checkpoint_directory
        self.hparam_results_directory = hparam_results_directory
        self.best_model_directory = best_model_directory
        self.model_performance_directory = model_perfromance_directory
        self.emulator_seed = emulator_seed
        self.study_seed = study_seed

        # create a list that will contain the evaluation metric for each training run
        self.evaluation_metric_list = []

    # define the objective function that optuna will use
    def objective(self, trial):
        # create the trainer
        emulator = MeanEmulator(model_class=MeanMLP,
                                model_hparams={'perceptrons_per_layer': trial.suggest_categorical("perceptrons_per_layer", self.hparam_tuning_dict['perceptrons_per_layer']),
                                               'n_dim': self.y_train_scaled.shape[1]},  
                                optimizer_class=optax.adamw,
                                optimizer_hparams={'learning_rate': trial.suggest_float(name='learning_rate',
                                                                                        low=self.hparam_tuning_dict[
                                                                                             'learning_rate']['low'],
                                                                                        high=self.hparam_tuning_dict[
                                                                                             'learning_rate']['high'],
                                                                                        log=self.hparam_tuning_dict[
                                                                                             'learning_rate']['log'])},
                                optimizer_schedule=True,
                                X_train=self.X_train_scaled,
                                y_train=self.y_train_scaled,
                                X_val=self.X_val_scaled,
                                y_val=self.y_val_scaled,
                                X_scaler_transform=self.scaler_X_transform,
                                y_scaler_inverse_transform=self.scaler_y_inverse_transform,
                                checkpoint_dir=self.checkpoint_directory,
                                seed=self.emulator_seed,
                                loss_fn=self.loss_fn,  
                                num_epochs=trial.suggest_categorical('num_epochs',
                                                                     self.hparam_tuning_dict['num_epochs']),
                                batch_size=self.batch_size)

        # train the model
        best_eval_metric = emulator.train()

        # save the model if it is the best one
        self.save_best_model(best_eval_metric, emulator)

        # delete the trainer to make sure we don't run into issues
        del emulator

        # return the best eval metric
        return best_eval_metric

    # define a function to save the best model
    def save_best_model(self, eval_metric, emulator):
        self.evaluation_metric_list.append(eval_metric)
        # check if the new eval metric is the smallest
        if eval_metric == min(self.evaluation_metric_list):
            # save plot of model training history
            emulator.plot_metrics_history(save_dir=self.model_performance_directory, prefix='mean_emulator_')

            # delete the old best model directory
            shutil.rmtree(self.best_model_directory,
                          ignore_errors=True)

            # and then copy over the checkpoint and hparam.json files to the best model directory
            src = self.checkpoint_directory
            dst = self.best_model_directory
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # the function that performs the hyperparameter tuning
    def tune_emulator(self):
        """
        Performs hyperparameter tuning for the mean emulator. Returns the best model loaded.
        """
        # print the hyperparameter tuning dictionary
        print('\nPreparing for hyperparameter tuning')
        print('The hyperparameters to tune are:')
        for key, value in self.hparam_tuning_dict.items():
            print(f'{key}: {value}')

        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = "study-mean"  # Unique identifier of the study.
        storage_name = f"sqlite:///{self.hparam_results_directory}/{study_name}.db"

        # create the study
        sampler = TPESampler(seed=self.study_seed)  
        study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name, storage=storage_name)
        study.optimize(self.objective, n_trials=self.ntrials, gc_after_trial=True)  

        # print out and save the results of the hparam study
        trial = study.best_trial
        print(f'\nBest Validation Loss: {trial.value}')
        print(f'Best Params:')
        for key, value in trial.params.items():
            print(f'-> {key}: {value}')
        print()

        # create dictionary to save the relevant information about the hyperparameter study
        hparam_study_results = {'hparam_space_explored': self.hparam_tuning_dict,
                                'number_of_trials': self.ntrials,
                                'best_params': trial.params,
                                'best_eval_metric': trial.values[0]}

        # Save hyperparameters
        with open(os.path.join(self.hparam_results_directory, 'hparam_study_results.json'), 'w') as f:
            json.dump(hparam_study_results, f, indent=4)

        # load the best model
        laf_mean_emulator = MeanEmulator.load_model(checkpoint_dir=self.best_model_directory,
                                                    X_train=self.X_train_scaled,
                                                    y_train=self.y_train_scaled,
                                                    X_val=self.X_val_scaled,
                                                    y_val=self.y_val_scaled,
                                                    X_scaler_transform=self.scaler_X_transform,
                                                    y_scaler_inverse_transform=self.scaler_y_inverse_transform)

        # delete the study to make sure we don't run into issues
        del study

        # return the best model
        return laf_mean_emulator


####################################################################################################
# a function to do hyperparameter tuning for the covariance emulator
####################################################################################################

class HParamTunerCovar:

    def __init__(self, hparam_tuning_dict, loss_fn, batch_size, ntrials, X_train_scaled, y_train_scaled, X_val_scaled,
                 y_val_scaled, scaler_X_transform, scaler_y_inverse_transform, checkpoint_directory, hparam_results_directory,
                 best_model_directory, model_performance_directory, emulator_seed=42, study_seed=10):
        """
        Does hyperparameter tuning for the mean emulator, saves the best model, and returns it.

        Args:
          hparam_tuning_dict: A dictionary containing the hyperparameters to be tuned.
          ntrials: The number of trials to run.
          loss_fn: The loss function used to train the emulator.
          batch_size: The batch size used during training.
          X_train_scaled: The input data scaled.
          y_train_scaled: The output data scaled.
          X_val_scaled: The input data scaled for the validation set.
          y_val_scaled: The output data scaled for the validation set.
          scaler_X_transform: The scaler used to scale the input data.
          scaler_y_inverse_transform: The scaler used to scale the output data.
          checkpoint_directory: The directory where the checkpoints are saved.
          hparam_results_directory: The directory where the hparam tuning results are saved.
          best_model_directory: The directory where the best model is saved.
          model_performance_directory: The directory where the model performance is saved.
          emulator_seed: The seed used for the emulator.
          study_seed: The seed used for the hyper-parameter study.

        Returns:
          the trained best emulator.
        """
        # save the input arguments
        self.hparam_tuning_dict = hparam_tuning_dict
        self.ntrials = ntrials
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.X_train_scaled = X_train_scaled
        self.y_train_scaled = y_train_scaled
        self.X_val_scaled = X_val_scaled
        self.y_val_scaled = y_val_scaled
        self.scaler_X_transform = scaler_X_transform
        self.scaler_y_inverse_transform = scaler_y_inverse_transform
        self.checkpoint_directory = checkpoint_directory
        self.hparam_results_directory = hparam_results_directory
        self.best_model_directory = best_model_directory
        self.model_performance_directory = model_performance_directory
        self.emulator_seed = emulator_seed
        self.study_seed = study_seed

        # create a list that will contain the evaluation metric for each training run
        self.evaluation_metric_list = []

    # define the objective function that optuna will use
    def objective(self, trial):
        # calculate the number of dimensions
        __d__ = self.y_train_scaled.shape[1]
        __n_dim__ = int((np.sqrt(1+8*__d__)-1)/2)

        # create the trainer
        trainer = CovarEmulator(model_class=CovarMLP,
                                model_hparams={'perceptrons_per_layer': trial.suggest_categorical("perceptrons_per_layer", self.hparam_tuning_dict['perceptrons_per_layer']),
                                               'n_dim': __n_dim__},
                                optimizer_class=optax.adamw,
                                optimizer_hparams={'learning_rate': trial.suggest_float(name='learning_rate',
                                                                                        low=self.hparam_tuning_dict['learning_rate']['low'],
                                                                                        high=self.hparam_tuning_dict['learning_rate']['high'],
                                                                                        log=self.hparam_tuning_dict['learning_rate']['log'])},
                                optimizer_schedule=True,
                                X_train=self.X_train_scaled,
                                y_train=self.y_train_scaled,
                                X_val=self.X_val_scaled,
                                y_val=self.y_val_scaled,
                                X_scaler_transform=self.scaler_X_transform,
                                y_scaler_inverse_transform=self.scaler_y_inverse_transform,
                                checkpoint_dir=self.checkpoint_directory,
                                seed=self.emulator_seed,
                                loss_fn=self.loss_fn,  
                                num_epochs=trial.suggest_categorical('num_epochs', self.hparam_tuning_dict['num_epochs']),
                                batch_size=self.batch_size)  

        # train the model
        best_eval_metric = trainer.train()

        # save the model if it is the best one
        self.save_best_model(best_eval_metric, trainer)

        # delete the trainer to make sure we don't run into issues
        del trainer
        # return the best eval metric
        return best_eval_metric

    # define a function to save the best model
    def save_best_model(self, eval_metric, trainer):
        self.evaluation_metric_list.append(eval_metric)
        # check if the new eval metric is the smallest
        if eval_metric == min(self.evaluation_metric_list):
            # save plot of model training history
            trainer.plot_metrics_history(save_dir=self.model_performance_directory, prefix='covar_emulator_')
            # delete the old best model directory
            shutil.rmtree(self.best_model_directory,
                          ignore_errors=True)

            # and then copy over the checkpoint and hparam.json files to the best model directory
            src = self.checkpoint_directory
            dst = self.best_model_directory
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # the function that performs the hyperparameter tuning
    def tune_emulator(self):
        """
        Performs hyperparameter tuning for the mean emulator. Returns the best model loaded.
        """
        # print the hyperparameter tuning dictionary
        print('\nPreparing for hyperparameter tuning')
        print('The hyperparameters to tune are:')
        for key, value in self.hparam_tuning_dict.items():
            print(f'{key}: {value}')

        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        study_name = "study-covar"  # Unique identifier of the study.
        storage_name = f"sqlite:///{self.hparam_results_directory}/{study_name}.db"

        # create the study
        sampler = TPESampler(seed=self.study_seed)  
        study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name, storage=storage_name)
        study.optimize(self.objective, n_trials=self.ntrials, gc_after_trial=True) 

        # print out and save the results of the hparam study
        trial = study.best_trial
        print(f'\nBest Validation Loss: {trial.value}')
        print(f'Best Params:')
        for key, value in trial.params.items():
            print(f'-> {key}: {value}')
        print()

        # create dictionary to save the relevant information about the hyperparameter study
        hparam_study_results = {'hparam_space_explored': self.hparam_tuning_dict,
                                'number_of_trials': self.ntrials,
                                'best_params': trial.params,
                                'best_eval_metric': trial.values[0]}

        # Save hyperparameters
        with open(os.path.join(self.hparam_results_directory, 'hparam_study_results.json'), 'w') as f:
            json.dump(hparam_study_results, f, indent=4)

        # load the best model
        emulator = CovarEmulator.load_model(checkpoint_dir=self.best_model_directory,
                                            X_train=self.X_train_scaled,
                                            y_train=self.y_train_scaled,
                                            X_val=self.X_val_scaled,
                                            y_val=self.y_val_scaled,
                                            X_scaler_transform=self.scaler_X_transform,
                                            y_scaler_inverse_transform=self.scaler_y_inverse_transform)

        # delete the study to make sure we don't run into issues
        del study

        # return the best model
        return emulator