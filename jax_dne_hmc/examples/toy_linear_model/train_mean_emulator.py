####################################################################################################
#
# Example on training the mean emulator for the toy linear model.
#
####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import optax

from jax_dne_hmc.data import ToyLinearCovLoader
from jax_dne_hmc.dne.architectures import MeanMLP
from jax_dne_hmc.dne.mean_emulator import MeanEmulator
from jax_dne_hmc.dne.scalers import DiffMinMaxScaler
from jax_dne_hmc.dne.losses import mape

# Directories for results
checkpoint_dir = '/Users/diegogonzalez/Documents/Research/ENIGMA/DNE-HMC/jax_dne_hmc/jax_dne_hmc/examples/toy_linear_model/mean_emulator_results/checkpoints'
results_dir = '/Users/diegogonzalez/Documents/Research/ENIGMA/DNE-HMC/jax_dne_hmc/jax_dne_hmc/examples/toy_linear_model/mean_emulator_results/results'


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

# split the dataset into training and testing
print('\nCreating shuffled training and test sets')
# now that we have a dataset, we need to separate it into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(theta,
                                                    mu,
                                                    test_size=0.2,
                                                    random_state=32)


# now that we have a dataset, we need to separate it into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.2,
                                                  random_state=32)

print('\nThe sizes for the datasets are:')
print(f'training set: {X_train.shape[0]}')
print(f'validation set: {X_val.shape[0]}')
print(f'test set: {X_test.shape[0]}')

# we can use a DiffStandardScaler or a DiffMinMaxScaler
scaler_X = DiffMinMaxScaler()  # scaler for input values
print(f'\nPreprocessing x data, using a {scaler_X.__class__.__name__}')
# fit the scaler ONLY TO THE TRAINING DATA
scaler_X.fit(X_train)
# transform the training dataset
X_train_scaled = scaler_X.transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = DiffMinMaxScaler()  # scaler for target values
print(f'\nPreprocessing y data, using a {scaler_y.__class__.__name__}')
# fit the scaler ONLY TO THE TRAINING DATA
scaler_y.fit(y_train)
# transform the training dataset
y_train_scaled = scaler_y.transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

print('\nPlotting all the mean models in the training set as an example')
# make another plot showing examples of the scaled data
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for mean_model in y_train[:]:
    ax.plot(x, mean_model, alpha=0.2)

ax.set_title('Mean observations in the training set', fontsize=17)
ax.set_ylabel('y', fontsize=17)
ax.set_xlabel('x', fontsize=17)
plt.show()

print('Plotting all the scaled mean models in the training set as an example')
# make another plot showing examples of the scaled data
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for mean_model in y_train_scaled[:]:
    ax.plot(x, mean_model, alpha=0.2)

ax.set_title('Scaled y in the training set', fontsize=17)
ax.set_ylabel('Scaled y', fontsize=17)
ax.set_xlabel('x', fontsize=17)
plt.show()

# ####################################################################################################

# now we can create a simple MLP model and save it
mean_trainer = MeanEmulator(model_class=MeanMLP,
                            model_hparams={'perceptrons_per_layer': [10, 10, 10],
                                           'n_dim': 11},  # number of dimensions of the observation
                            optimizer_class=optax.adamw,
                            optimizer_hparams={'learning_rate': 0.01},
                            optimizer_schedule=True,
                            X_train=X_train_scaled,
                            y_train=y_train_scaled,
                            X_val=X_val_scaled,
                            y_val=y_val_scaled,
                            X_scaler_transform=scaler_X.transform,
                            y_scaler_inverse_transform=scaler_y.inverse_transform,
                            checkpoint_dir=checkpoint_dir,
                            seed=42,
                            loss_fn=mape,
                            patience=200,
                            num_epochs=1000,
                            batch_size=32)

best_eval_metrics = mean_trainer.train()
# plot the training and validation losses
mean_trainer.plot_metrics_history(save_dir=results_dir, prefix='mean_emu_')

# load the best model, and bind it to do predictions
mean_emulator = MeanEmulator.load_model(checkpoint_dir=checkpoint_dir,
                                            X_train=X_train,
                                            y_train=y_train_scaled,
                                            X_val=X_val,
                                            y_val=y_val_scaled,
                                            X_scaler_transform=scaler_X.transform,
                                            y_scaler_inverse_transform=scaler_y.inverse_transform)

model_bd = mean_emulator.bind_model()

# compare the bound model predictions with my own predict function
predicted_y_scaled_bound = model_bd(X_train_scaled)
predicted_y_unscaled_bound = mean_emulator.y_scaler_inverse_transform(predicted_y_scaled_bound)
predicted_y = mean_emulator.predict(X_unscaled=X_train)

print(f'\nbound model scaled predictions shape: {predicted_y_scaled_bound.shape}')
print(f'bound model predictions shape: {predicted_y_unscaled_bound.shape}')
print(f'my implementation of predictions shape: {predicted_y.shape}')
print(f'np.allclose is {np.allclose(predicted_y_unscaled_bound, predicted_y)}\n')

# calculate the mean absolute percentage error in the test set and print it out
predicted_y_test = mean_emulator.predict(X_unscaled=X_test)
mape_test = mape(y_test, predicted_y_test)

print(f'The mean absolute percentage error in the test set is {mape_test.mean()}')
