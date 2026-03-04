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

from jax_dne_hmc.data import ToyLinearCovLoader
from jax_dne_hmc.dne.architectures import CovarMLP
from jax_dne_hmc.dne.covariance_emulator import CovarEmulator
from jax_dne_hmc.dne.scalers import DiffMinMaxScaler
from jax_dne_hmc.dne.losses import mape

from IPython import embed

# Directories for results
checkpoint_dir = '/Users/diegogonzalez/Documents/Research/ENIGMA/DNE-HMC/jax_dne_hmc/jax_dne_hmc/examples/toy_linear_model/covar_emulator_results/checkpoints'
results_dir = '/Users/diegogonzalez/Documents/Research/ENIGMA/DNE-HMC/jax_dne_hmc/jax_dne_hmc/examples/toy_linear_model/covar_emulator_results/results'


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

#################################################################

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
                              checkpoint_dir=checkpoint_dir,
                              seed=42,
                              loss_fn=mape,
                              patience=200,
                              num_epochs=1000,
                              batch_size=32)

best_eval_metrics = covar_trainer.train()
# plot the training and validation losses
covar_trainer.plot_metrics_history(save_dir=results_dir, prefix='covar_emu_')

# load the best model, and bind it to do predictions
covar_emulator = CovarEmulator.load_model(checkpoint_dir=checkpoint_dir,
                                          X_train=X_train,
                                          y_train=y_covar_train_scaled,
                                          X_val=X_val,
                                          y_val=y_covar_val_scaled,
                                          X_scaler_transform=scaler_X.transform,
                                          y_scaler_inverse_transform=scaler_y_covar.inverse_transform)
