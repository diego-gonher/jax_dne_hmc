![DNE + HMC logo](DNE%2BHMC%20logo.png)
# DNE + HMC: Differentiable Neural Emulators with Hamiltonian Monte Carlo

Implementation of **Differentiable Neural Emulators (DNE) with Hamiltonian Monte Carlo (HMC)** for parameter estimation in problems where the observable is a **1D vector** (e.g., power spectra, correlation functions) that follows a **multivariate Gaussian likelihood with model-dependent covariance**.

This repository provides a clean implementation of the inference method introduced in:

**González-Hernández et al. (2025)**
https://arxiv.org/abs/2509.13498

The goal of this repository is to provide a **standalone, easy-to-use implementation** of the method so that other researchers can apply it to their own problems.

---

# Method Overview

Many scientific inference problems involve summary statistics that can be modeled as

$$
\mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}(\theta), \Sigma(\theta))
$$

where

* $\mathbf{y}$ is the observed data vector (e.g., power spectrum)
* $\theta$ are the physical model parameters
* $\boldsymbol{\mu}(\theta)$ is the model prediction
* $\Sigma(\theta)$ is the **parameter-dependent covariance matrix**

Direct evaluation of $\mu(\theta)$ and $\Sigma(\theta)$ may require **expensive simulations**. The approach implemented here replaces them with **neural emulators**.

Two neural networks are trained:

### Mean Emulator

Learns

$$
\theta \rightarrow \mu(\theta)
$$

### Covariance Emulator

Learns the **Cholesky factor** of the covariance matrix

$$
\theta \rightarrow L(\theta)
$$

where

$$
\Sigma(\theta) = L(\theta)L(\theta)^T
$$

To guarantee **positive definiteness**, the emulator predicts a **flattened Cholesky factor**

* diagonal elements are stored as $\log L_{ii}$
* lower triangular elements are stored directly

After emulation, the likelihood becomes

$$
\mathcal{L}(\theta) =
\mathcal{N}\left(\mathbf{y}_{obs}\mid \mu(\theta), \Sigma(\theta)\right)
$$

and parameter inference is performed using **Hamiltonian Monte Carlo (HMC)**.

A schematic of the workflow:

```
Simulations → Summary statistics
        ↓
Train μ emulator and Σ emulator
        ↓
Likelihood with emulated μ(θ), Σ(θ)
        ↓
Hamiltonian Monte Carlo
        ↓
Posterior samples
```

---

# Installation

The repository uses a standard Python packaging setup.

Clone the repository:

```bash
git clone https://github.com/<your-repo>/jax_dne_hmc.git
cd jax_dne_hmc
```

Install in editable mode:

```bash
pip install -e .
```

This installs the package and all dependencies defined in `pyproject.toml`.

> [!WARNING]
> Installing JAX can be difficult in certain systems, making the use of this pyproject.toml difficult. If you are having issues, I recommend you create a clean python environment, install a workable version of JAX, and then try installing this repository.

---

# Example Dataset

The repository includes a **toy dataset** based on a simple linear model:

$$
y = m x + b
$$

with parameters

$$
\theta = (m, b)
$$

For each parameter pair:

* the mean vector is the noiseless model
* observations are drawn from a multivariate Gaussian
* the covariance matrix is parameter dependent

Dataset structure:

```
theta    (N_models, N_params)
mu       (N_models, N_bins)
Sigma    (N_models, N_bins, N_bins)
y_mocks  (N_models, N_mocks, N_bins)
x        (N_bins)
```

You can find a complete explanation of this mode in the example's [README](jax_dne_hmc/examples/toy_linear_model/README.md).

---

# Training the Mean Emulator

A minimal example is provided in:

```
examples/toy_linear_model/train_mean_emulator.py
```

Example usage:

```python
from jax_dne_hmc.data import ToyLinearCovLoader
from jax_dne_hmc.dne.mean_emulator import MeanEmulator
from jax_dne_hmc.dne.architectures import MeanMLP

loader = ToyLinearCovLoader()
data = loader.get_data()

theta = data["theta"]
mu = data["mu"]

mean_trainer = MeanEmulator(
    model_class=MeanMLP,
    model_hparams={
        "perceptrons_per_layer": [10,10,10],
        "n_dim": 11
    },
)

mean_trainer.train()
```

The emulator learns

$$
\theta \rightarrow \mu(\theta)
$$

---

# Training the Covariance Emulator

The covariance emulator predicts the **flattened Cholesky factor** of the covariance matrix.

Example script:

```
examples/toy_linear_model/train_covar_emulator.py
```

Basic usage:

```python
from jax_dne_hmc.dne.covariance_emulator import CovarEmulator
from jax_dne_hmc.dne.architectures import CovarMLP

covar_trainer = CovarEmulator(
    model_class=CovarMLP,
    model_hparams={
        "perceptrons_per_layer":[50,50,50,50],
        "n_dim":11
    }
)

covar_trainer.train()
```

The network learns

$$
\theta \rightarrow L(\theta)
$$

from which the covariance is reconstructed as

$$
\Sigma(\theta)=L(\theta)L(\theta)^T
$$

---

# Hyper-Parameter Tuning

Both emulators include an **automatic hyper-parameter tuner**.

Example:

```python
from jax_dne_hmc.dne.hparam_tuning import HParamTunerMean

tuner = HParamTunerMean(
    hparam_tuning_dict={
        "perceptrons_per_layer":[[10,10,10],[20,20,20]],
        "learning_rate":{"low":1e-4,"high":1e-1},
        "num_epochs":[500,1000]
    }
)

best_model = tuner.tune_emulator()
```

A similar interface exists for the covariance emulator:

```python
HParamTunerCovar
```

The tuner automatically:

* samples hyper-parameters
* trains models
* evaluates validation loss
* saves the best emulator

---

# Using the HMC Inference Class

Once mean and covariance emulators are trained (for example with the toy linear model scripts),
you can run HMC inference using `HMCInference`, as illustrated in
`jax_dne_hmc/examples/toy_linear_model/inference_script.py`:

```python
from jax_dne_hmc.data import ToyLinearCovLoader
from jax_dne_hmc.dne.mean_emulator import MeanEmulator
from jax_dne_hmc.dne.covariance_emulator import CovarEmulator
from jax_dne_hmc.hmc.hmc import HMCInference

loader = ToyLinearCovLoader()
data = loader.get_data()
theta, mu, Sigma, mocks = data["theta"], data["mu"], data["Sigma"], data["y_mocks"]

# load best-trained emulators (paths follow the training examples)
mean_emu = MeanEmulator.load_model(checkpoint_dir=".../mean_best_model", ...)
cov_emu = CovarEmulator.load_model(checkpoint_dir=".../covar_best_model", ...)

theta_ranges = [(3.0, 8.0), (1.0, 4.0)]  # example priors on (m, b)
hmc = HMCInference(theta_astro_ranges=theta_ranges,
                   laf_mean_emulator=mean_emu,
                   laf_cov_emulator=cov_emu,
                   dataset_loader=None,
                   z_ti=None)

x_opt, theta_opt, _ = hmc.fit_one(autocorrelation_fn_data=mocks[0].mean(axis=0))
results = hmc.mcmc_one(key=hmc.key, x_opt=x_opt, autocorrelation_fn_data=mocks[0].mean(axis=0))
```

The example script in `examples/toy_linear_model/inference_script.py` shows a complete pipeline:
loading trained emulators, selecting a subset of mocks, running HMC, and saving diagnostics.

---

# Repository Structure

```
jax_dne_hmc
│
├── jax_dne_hmc
│   ├── data            # dataset loaders and toy HDF5 access
│   ├── dne             # architectures, emulators, scalers, losses, hparam tuning
│   ├── hmc             # HMC inference classes
│   ├── utils           # covariance and HMC utility functions
│   └── examples
│       └── toy_linear_model  # training, tuning, and inference scripts
│
├── pyproject.toml
└── README.md
```

---

# Status

Current implemented features:

* Mean emulator training
* Covariance emulator training
* Hyper-parameter tuning utilities
* HMC inference pipeline
* Toy dataset and full training examples

Planned additions:

* posterior diagnostics

---

# Citation

If you use this repository, please cite:

```
González-Hernández et al. (2025)
Differentiable Neural Emulators with Hamiltonian Monte Carlo
https://arxiv.org/abs/2509.13498
```

---

# Contact

Questions or suggestions are welcome.
Feel free to open an issue or contact the author.
