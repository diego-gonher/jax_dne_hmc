####################################################################################################
#
# Example on training the mean emulator for the toy linear model.
#
####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import jax
import jax.numpy as jnp
import optax
import os

from jax_dne_hmc.data import ToyLinearCovLoader
from jax_dne_hmc.dne.architectures import CovarMLP
from jax_dne_hmc.dne.covariance_emulator import CovarEmulator
from jax_dne_hmc.dne.scalers import DiffMinMaxScaler
from jax_dne_hmc.dne.losses import mape
from jax_dne_hmc.dne.hparam_tuning import HParamTunerCovar
from jax_dne_hmc.utils.covar_metrics import calculate_cov_error_optimized

from IPython import embed

####################################################################################################
#                                     PIPELINE PARAMETERS                                          #
####################################################################################################

# Directories for results
base_directory = '/Users/diegogonzalez/Documents/Research/ENIGMA/DNE-HMC/jax_dne_hmc/jax_dne_hmc/examples/toy_linear_model/covar_emulator_results'
covar_checkpoint_dir = f'{base_directory}/checkpoints'
covar_hparam_results_dir = f'{base_directory}/hparam_results'
covar_best_model_directory = f'{covar_hparam_results_dir}/best_model'
covar_emulation_error_directory = f'{base_directory}/emulation_metrics'

os.makedirs(covar_checkpoint_dir, exist_ok=True)
os.makedirs(covar_hparam_results_dir, exist_ok=True)
os.makedirs(covar_best_model_directory, exist_ok=True)
os.makedirs(covar_emulation_error_directory, exist_ok=True)

# Hyper-parameter tuning parameters
N_TRIALS_COVAR = 10  # you might need to reduce this in case you don't have enough memory on your machine

# hparam dictionary for the covar emulator
# hparam dictionary for the covar emulator
covar_hparam_tuning_dict = {'perceptrons_per_layer': [[50, 50, 50, 50],
                                                      [25, 25, 25, 25],
                                                      [25, 25, 50, 50]],
                            'learning_rate': {'low': 1e-7,
                                              'high': 1e0,
                                              'log': False},
                            'num_epochs': [750, 1000, 1250, 1500, 1750]}

####################################################################################################
#                                    LOAD AND PREPARE DATA                                         #
####################################################################################################

# Load the toy dataset (default path: package's data/datasets/toy_linear_cov_dataset.h5)
loader = ToyLinearCovLoader()
data = loader.get_data()

# Unpack for later use: theta (params), mu (mean), Sigma (cov), y_mocks (observations), x (design)
theta = data["theta"]  
mu = data["mu"]
Sigma = data["Sigma"]
mocks = data["y_mocks"]
x = data["x"]

# print message with shapes
print(f'Dataset loaded:')
print(f'- theta shape: {theta.shape}')
print(f'- mu shape: {mu.shape}')
print(f'- Sigma shape: {Sigma.shape}')
print(f'- mocks shape: {mocks.shape}')
print(f'- x shape: {x.shape}')

# Get the flattened Cholesky decomposition of the covariance matrices
# first, we need to get the number of parameters
N_PARAMS = theta.shape[1]
# second, we need to get the number of bins
N_BINS = mu.shape[1]
# third, we need to get the number of data points
N_DATA_POINTS = theta.shape[0]

# create an array to store the flattened Cholesky decomposition of the covariance matrices
# get the cholesky factors, first create the two empty arrays
y_cholesky_factors = np.zeros((N_DATA_POINTS, N_BINS, N_BINS))
y_cholesky_factors_flat = np.zeros((N_DATA_POINTS, int(N_BINS * (N_BINS + 1) / 2)))
# get the indices of the lower triangular elements
# diagonal elements
d_idx = jnp.diag_indices(n=N_BINS)
# lower triangular indices
l_idx = jnp.tril_indices(n=N_BINS, k=-1)

# we have to filter out problematic models with negative eigenvalues, we will store the indices of the good models
good_models_idx = []
bad_models_idx = []

# loop over all the models and fill the arrays
for i in range(N_DATA_POINTS):
    # get the covariance matrix for this parameter pair
    covariance_i = Sigma[i, :, :]
    # get the Cholesky decomposition of the covariance matrix
    # calculate the lower triangular Cholesky factor
    cholesky_factor_i = jax.scipy.linalg.cholesky(a=covariance_i, lower=True)  # jax.
    # save the Cholesky factor
    y_cholesky_factors[i, :, :] = cholesky_factor_i
    # save the flattened Cholesky factor, notice the ln in the diagonal elements
    y_cholesky_factors_flat[i, :N_BINS] = np.log(cholesky_factor_i[d_idx])
    y_cholesky_factors_flat[i, N_BINS:] = cholesky_factor_i[l_idx]

    # check if there are nan values in the Cholesky factor
    if np.any(np.isnan(y_cholesky_factors_flat[i, :])):
        print(f'Nan values in model {i}, with params {theta[i, :]}')
        bad_models_idx.append(i)
    else:
        good_models_idx.append(i)

# embed(header='check the arrays')

# filter the models with negative eigenvalues
theta = theta[good_models_idx, :]
mu = mu[good_models_idx, :]
Sigma = Sigma[good_models_idx, :, :]
y_cholesky_factors = y_cholesky_factors[good_models_idx, :, :]
y_cholesky_factors_flat = y_cholesky_factors_flat[good_models_idx, :]

# split the dataset into training and testing
print('\nCreating shuffled training and test sets')
# now that we have a dataset, we need to separate it into training and testing sets
X_train, X_test, y_mean_train, y_mean_test, y_covar_train, y_covar_test = train_test_split(theta,
                                                                                           mu,
                                                                                           y_cholesky_factors_flat,
                                                                                           test_size=0.2,
                                                                                           random_state=32)

# create validation sets
X_train, X_val, y_mean_train, y_mean_val, y_covar_train, y_covar_val = train_test_split(X_train,
                                                                                        y_mean_train,
                                                                                        y_covar_train,
                                                                                        test_size=0.2,
                                                                                        random_state=32)
                                                                                    
print('\nThe sizes for the datasets are:')
print(f'training set: {X_train.shape[0]}')
print(f'validation set: {X_val.shape[0]}')
print(f'test set: {X_test.shape[0]}')

# do the data scaling
# first scale the x data, which is shared on both emulators
scaler_X = DiffMinMaxScaler()  # scaler for input values
print(f'\nPreprocessing x data, using a {scaler_X.__class__.__name__}')
# fit the scaler ONLY TO THE TRAINING DATA
scaler_X.fit(X_train)
# transform the training dataset
X_train_scaled = scaler_X.transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# now we scale both y datasets
# first the mean  DiffStandardScaler  DiffMinMaxScaler
scaler_y_mean = DiffMinMaxScaler()  # scaler for target values
print(f'\nPreprocessing y data for the mean emulator, using a {scaler_y_mean.__class__.__name__}')
# fit the scaler ONLY TO THE TRAINING DATA
scaler_y_mean.fit(y_mean_train)
# transform the training dataset
y_mean_train_scaled = scaler_y_mean.transform(y_mean_train)
y_mean_val_scaled = scaler_y_mean.transform(y_mean_val)
y_mean_test_scaled = scaler_y_mean.transform(y_mean_test)

# now the covariance  DiffStandardScaler  DiffMinMaxScaler
scaler_y_covar = DiffMinMaxScaler()  # scaler for target values
print(f'\nPreprocessing y data for the covariance emulator, using a {scaler_y_covar.__class__.__name__}')
# fit the scaler ONLY TO THE TRAINING DATA
scaler_y_covar.fit(y_covar_train)
# transform the training dataset
y_covar_train_scaled = scaler_y_covar.transform(y_covar_train)
y_covar_val_scaled = scaler_y_covar.transform(y_covar_val)
y_covar_test_scaled = scaler_y_covar.transform(y_covar_test)

# make a plot showing examples of the raw data
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

for i in range(y_covar_train.shape[0]):
    ax.plot(y_covar_train[i].ravel(), alpha=0.2)

ax.set_title('Flat Cholesky factors of covariance matrices in the training set', fontsize=17)
ax.set_ylabel('Power', fontsize=17)
ax.set_xlabel('Matrix element', fontsize=17)
plt.show()

# make a plot showing examples of the scaled data
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

for i in range(y_covar_train_scaled.shape[0]):
    ax.plot(y_covar_train_scaled[i].ravel(), alpha=0.2)

ax.set_title('Flat Cholesky factors of covariance matrices in the training set', fontsize=17)
ax.set_ylabel('Scaled Power', fontsize=17)
ax.set_xlabel('Matrix element', fontsize=17)
plt.show()

####################################################################################################
#                              SINGLE TRAIN OF THE COVAR EMULATOR                                  #
####################################################################################################

# now we can create a simple MLP model and save it
covar_trainer = CovarEmulator(model_class=CovarMLP,
                              model_hparams={'perceptrons_per_layer': [50, 50, 50, 50],
                                             'n_dim': 11},  # number of dimensions of the observation
                              optimizer_class=optax.adamw,
                              optimizer_hparams={'learning_rate': 0.01},
                              optimizer_schedule=True,
                              X_train=X_train_scaled,
                              y_train=y_covar_train_scaled,
                              X_val=X_val_scaled,
                              y_val=y_covar_val_scaled,
                              X_scaler_transform=scaler_X.transform,
                              y_scaler_inverse_transform=scaler_y_covar.inverse_transform,
                              checkpoint_dir=covar_checkpoint_dir,
                              seed=42,
                              loss_fn=mape,
                              patience=200,
                              num_epochs=1000,
                              batch_size=32)

best_eval_metrics = covar_trainer.train()
# plot the training and validation losses
covar_trainer.plot_metrics_history(save_dir=covar_hparam_results_dir, prefix='covar_emu_')

# load the best model, and bind it to do predictions
covar_emulator = CovarEmulator.load_model(checkpoint_dir=covar_checkpoint_dir,
                                          X_train=X_train,
                                          y_train=y_covar_train_scaled,
                                          X_val=X_val,
                                          y_val=y_covar_val_scaled,
                                          X_scaler_transform=scaler_X.transform,
                                          y_scaler_inverse_transform=scaler_y_covar.inverse_transform)


####################################################################################################
#                                 HPARAM TUNE THE COVAR EMULATOR                                   #
####################################################################################################
# create the hyperparameter tuner for the mean emulator
hparam_tuner_covar = HParamTunerCovar(hparam_tuning_dict=covar_hparam_tuning_dict,
                                      loss_fn=mape,
                                      batch_size=32,
                                      ntrials=N_TRIALS_COVAR,
                                      X_train_scaled=X_train_scaled,  # X_train_scaled  y_mean_train_scaled
                                      y_train_scaled=y_covar_train_scaled,
                                      X_val_scaled=X_val_scaled,  # X_val_scaled  y_mean_val_scaled
                                      y_val_scaled=y_covar_val_scaled,
                                      scaler_X_transform=scaler_X.transform,  # scaler_X.transform  scaler_y_mean
                                      scaler_y_inverse_transform=scaler_y_covar.inverse_transform,
                                      checkpoint_directory=covar_checkpoint_dir,
                                      hparam_results_directory=covar_hparam_results_dir,
                                      best_model_directory=covar_best_model_directory,
                                      model_performance_directory=covar_emulation_error_directory,
                                      emulator_seed=42,
                                      study_seed=10)

# close all figures
plt.close('all')

# run the hyperparameter tuning and get the best emulator for the mean
covar_emulator = hparam_tuner_covar.tune_emulator()

# produce the plots that serve to check the emulation error budget
list_of_covar_emulation_strings = []

# calculate the error on the diagonal of the covariance matrix as defined by Joe
# first, reconstruct the covariance matrices from the flat cholesky factors in the test set, this is the ground truth
true_test_covariance_matrices = covar_emulator.create_covariance_matrix(
        flat_cholesky_factor_unscaled=covar_emulator.y_scaler_inverse_transform(y_covar_test_scaled))

# now, predict the covariance matrices from the emulator
predicted_test_covariance_matrices = covar_emulator.predict(X_unscaled=X_test)


emulated_covar_errors = np.zeros((true_test_covariance_matrices.shape[0],
                                  true_test_covariance_matrices.shape[1],
                                  true_test_covariance_matrices.shape[2]))
emulated_covar_errors_diag = np.zeros((true_test_covariance_matrices.shape[0],
                                       true_test_covariance_matrices.shape[1]))

# embed(header='Loaded the covariance matrix emulator')

# loop through the test set and calculate the error for each covariance matrix, both the diagonal and entire
for i in range(len(true_test_covariance_matrices)):
    # calculate the error for the entire covariance matrix
    error_i = calculate_cov_error_optimized(true_covariance_matrix=true_test_covariance_matrices[i],
                                            emulated_covariance_matrix=predicted_test_covariance_matrices[i])
    # assign the error to the array, and its diagonal as well
    emulated_covar_errors[i] = error_i
    emulated_covar_errors_diag[i] = np.diag(emulated_covar_errors[i])


print('\nThe diagonal error')
# make another plot showing examples of the scaled data
with plt.style.context(['science', 'no-latex']):
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.75))
    ax.plot(x, emulated_covar_errors_diag.mean(axis=0), alpha=0.7, color='red')
    ax.set_title('Mean diag(covar) error')
    ax.set_ylabel('Error')
    ax.set_xlabel('log $k_{eff}$')
    plt.savefig(os.path.join(covar_emulation_error_directory, 'diag_covar_error.pdf'), bbox_inches='tight', dpi=300)
