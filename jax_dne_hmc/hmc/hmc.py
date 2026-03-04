####################################################################################################
#
# This module has a class to do inference using our neural emulators and HMC sampling.
#
####################################################################################################

import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import jit, random
import jax.numpy as jnp
from jax import config

from numpyro.infer import MCMC, NUTS
import arviz as az
import h5py
import optax
from tqdm.auto import trange
# from scipy.optimize import minimize

from functools import partial
import time
from IPython import embed

from qso_fitting.fitting.utils import bounded_theta_to_x, x_to_bounded_theta  # variable transformations
from qso_fitting.fitting.utils import bounded_variable_lnP  # prior

# from dw_inference.inference.utils import walker_plot, corner_plot

# use 64 bit precision
config.update("jax_enable_x64", True)

# TESTING Disable JIT for now
# from jax.config import config
# config.update('jax_disable_jit', True)
# TESTING

# embed(header='Check the imports')


# A class to do the inference
class HMCInference:
    """
    A class to do the inference using HMC sampling. Assumptions include a multivariate gaussian likelihood function and uniform, rectangular priors.

    """

    def __init__(self, theta_astro_ranges, laf_mean_emulator, laf_cov_emulator, dataset_loader, z_ti,
                 opt_nsteps=150, opt_lr=0.01, mcmc_nsteps=1000, mcmc_num_chains=4, mcmc_warmup=1000,
                 mcmc_init_perturb=0.05,  mcmc_max_tree_depth=10, mcmc_dense_mass=True, key=random.PRNGKey(42)):
        """
        Initialize the HMCInference class.

        Args:
            theta_astro_ranges (list):
                List of length n_params containing 2-d tuples, where each tuple is the range of the parameter.
                The first element of the tuple is the lower bound, and the second element is the upper bound.
            laf_mean_emulator (TrainerModule):
                An instance of the LAFMeanEmulator class.
            laf_cov_emulator (LAFDatasetLoader):
                The emulator used for the covariance of the autocorrelation function. Currently, I use the dataset
                loader as a placeholder until I have the emulator ready.
            dataset_loader (LAFDatasetLoader):
                An instance of the LAFDatasetLoader class. This is used alognside true thetas and with NGP interpolation
            z_ti (String):
                String corresponding to the redshift bin of the model to infer, this is temporary.
            mfp_ti (float):
                The mean free path of the model to infer, this is temporary.
            mean_flux_ti (float):
                The mean flux of the model to infer, this is temporary.
            opt_nsteps (int):
                Number of the quick optimization steps used for initializing x for the HMC.
            opt_lr (float):
                Learning rate for the quick optimization steps used for initializing x for the HMC.
            mcmc_nsteps (int):
                number of steps for MCMC per chain. Total steps is this number times mcmc_num_chains
            mcmc_num_chains (int):
                number of MCMC chains
            mcmc_warmup (int):
                number of warmup steps for MCMC per chain
            mcmc_init_perturb (float):
                fractional amount to perturb about optimum for MCMC chains
            mcmc_max_tree_depth (int):
                max depth of the binary tree created during the doubling scheme of NUTS HMC sampler
            mcmc_dense_mass (bool):
                flag indicating if mass matrix is dense or diagonal
            key (JAX PRNG key):
                pseudo-random number generator key.
        """
        # STORE THE PARAMETERS TO INFER, THIS IS TEMPORARY AND ONLY USED BY THE mcmc METHOD
        self.z_ti = z_ti

        # Store the parameters
        self.ndim = 2  # this is hardcoded for our case
        self.theta_astro_ranges = theta_astro_ranges
        self.theta_astro_inits = tuple([np.mean([tup[0], tup[1]]) for tup in theta_astro_ranges])  # mean of the ranges
        self.theta_astro_mins = jnp.array([astro_par_range[0] for astro_par_range in self.theta_astro_ranges])
        self.theta_astro_maxs = jnp.array([astro_par_range[1] for astro_par_range in self.theta_astro_ranges])

        # Set astro parameter priors (Smoothed Box Prior is a differentiable uniform distribution)
        self.x_astro_priors = [bounded_variable_lnP, bounded_variable_lnP]

        # Store the emulators and the dataset loader
        self.laf_mean_emulator = laf_mean_emulator
        self.laf_cov_emulator = laf_cov_emulator
        self.dataset_loader = dataset_loader

        # Set the optimizer parameters
        self.opt_nsteps = opt_nsteps
        self.opt_lr = opt_lr

        # Set the MCMC parameters
        self.mcmc_nsteps = mcmc_nsteps  # Number of steps
        self.mcmc_num_chains = mcmc_num_chains  # Number of walkers
        self.mcmc_warmup = mcmc_warmup  # Number of burning steps
        self.mcmc_max_tree_depth = mcmc_max_tree_depth
        self.mcmc_dense_mass = mcmc_dense_mass
        self.mcmc_init_perturb = mcmc_init_perturb  # fractional amount to perturb about optimum for MCMC chains
        self.key = key  # random number generator key

        # These are some class attributes used in the case of multiple inferences that are used to save the results
        self.n_mock_datasets = None
        self.theta_true = None
        self.theta_true_ngp = None  # THIS IS TEMPORARY
        self.theta_true_og = None  # THIS IS TEMPORARY
        self.gaussian_mocks = None  # this is used to save the correct true model, which depends on the type of mocks
        self.autocorrelation_fn_mock_datasets = None
        self.x_true = None
        self.mcmc_nsteps_tot = None
        self.neff = None
        self.neff_mean = None
        self.r_hat = None
        self.r_hat_mean = None
        self.hmc_num_steps = None
        self.hmc_tree_depth = None
        self.sec_per_neff = None
        self.ms_per_step = None
        self.runtime = None
        self.samples = None
        self.x_samples = None
        self.ln_probs_x = None
        # Some other things we need if truths were provided
        self.ln_prob_x_true = None
        self.ln_like_true = None
        self.ln_prior_x_true = None

    # functions to transform between theta and x
    # x to theta
    @partial(jit, static_argnums=(0,))
    def x_to_theta(self, x):
        """
        Transform from HMC parameter vector x into parameter vector theta. This is the vectorized version of the
        function _x_to_theta below.

        Args:
            x (jax.numpy array):
               HMC parameter vector. Shape (nsamples, ndim) or (ndim,)
        Returns:
            theta (jax.numpy array):
               Parameter vector. Shape (nsamples, ndim) or (ndim,)
        """
        theta = jax.vmap(self._x_to_theta, in_axes=0, out_axes=0)(jnp.atleast_2d(x))
        return theta.squeeze()

    @partial(jit, static_argnums=(0,))
    def _x_to_theta(self, x):
        """
        Transform from HMC parameter vector x into parameter vector theta. This is the single element version called by
         the vectorized version above.

        Args:
            x (jax.numpy array):
               HMC parameter vector. Shape (ndim,)
        Returns:
            theta (jax.numpy array):
               Parameter vector. Shape (ndim,)
        """

        theta = x_to_bounded_theta(x, self.theta_astro_ranges)
        return theta

    # theta to x
    @partial(jit, static_argnums=(0,))
    def theta_to_x(self, theta):
        """
         Transform from parameter vector theta into the HMC parameter vector theta. This is the vectorized version of
         the function _theta_to_x below.

         Args:
             theta (jax.numpy array):
                Parameter vector. Shape (nsamples, ndim) or (ndim,)

         Returns:
             x (jax.numpy array):
                HMC parameter vector. Shape (nsamples, ndim) or (ndim,)
         """
        x = jax.vmap(self._theta_to_x, in_axes=0, out_axes=0)(jnp.atleast_2d(theta))
        return x.squeeze()

    @partial(jit, static_argnums=(0,))
    def _theta_to_x(self, theta):
        """
        Transform from parameter vector x into parameter vector theta. This is the single element version called by
        the vectorized version above.

        Args:
             theta (jax.numpy array):
                Parameter vector. Shape (nsamples, ndim) or (ndim,)

        Returns:
             x (jax.numpy array):
                HMC parameter vector. Shape (nsamples, ndim) or (ndim,)
        """
        x = bounded_theta_to_x(theta, self.theta_astro_ranges)
        return x

    # Now the log functions for the prior and likelihood evaluations
    @partial(jit, static_argnums=(0,))
    def ln_prior(self, x_astro):
        """
        Compute the prior on astro_params

        Args:
            x_astro (ndarray): shape = (nastro,)
                dimensionless astrophysical parameter vector

        Returns:
            prior (float):
                Prior on these model parameters
        """

        prior = 0.0
        for x_ast, x_astro_pri in zip(x_astro, self.x_astro_priors):
            prior += x_astro_pri(x_ast)

        return prior

    # function that uses the emulators to get the mean autocorrelation function and its corresponding covariance matrix
    @partial(jit, static_argnums=(0,))
    def get_mean_and_covar(self, theta_astro):
        """
        Returns the mean autocorrelation function and its corresponding covariance matrix given the astrophysical
        parameters.

        Args:
            theta_astro (array): theta_astro[0] is mfp, theta_astro[1] is mean_flux in physical units.

        Returns:
            (mean_autocorrelation_function, covariance_matrix)
        """
        # get the mean autocorrelation function from the neural emulator
        mean_autocorrelation_function = self.laf_mean_emulator.predict(theta_astro).ravel()  # Is ravel is needed?
        # get the covariance matrix from the neural emulator
        covariance_matrix = self.laf_cov_emulator.predict_single(theta_astro)

        return mean_autocorrelation_function, covariance_matrix

    # Gaussian likelihood
    @partial(jit, static_argnums=(0,))
    def ln_gaussian_likelihood(self, x_astro, autocorrelation_fn_data):
        """
        Natural logarithm of the gaussian likelihood.

        Args:
            x_astro (jnp.array): x_astro[0] is mfp, x_astro[1] is mean_flux in dimensionless units.
            autocorrelation_fn_data (jnp.array): autocorrelation function that the inference is being done for.

        Returns:
            float: ln of gaussian likelihood.
        """
        # convert to physical units
        theta_astro = self.x_to_theta(x_astro)
        # obtain the mean autocorrelation model and its corresponding covariance matrix
        mean_autocorrelation_function, covariance_matrix = self.get_mean_and_covar(theta_astro=theta_astro)
        # compute and return the gaussian likelihood
        return jax.scipy.stats.multivariate_normal.logpdf(x=autocorrelation_fn_data,
                                                          mean=mean_autocorrelation_function,
                                                          cov=covariance_matrix)

    # Now the potential function and its helper used for the numpyro NUTS sampler
    @partial(jit, static_argnums=(0,))
    def potential_fn(self, x_astro, autocorrelation_fn_data):
        """
            Potential function for the MCMC.

            Args:
                x_astro (array): x_astro[0] is mfp, x_astro[1] is mean_flux in dimensionless units.
                autocorrelation_fn_data (array): autocorrelation function that the inference is being done for.

            Returns:
                float: potential.
            """
        # ln(prior) calculated
        lnPrior = self.ln_prior(x_astro)
        # ln(likelihood) calculated
        lnL = self.ln_gaussian_likelihood(x_astro, autocorrelation_fn_data)
        # Calculate the potential = -(ln(likelihood) + ln(prior)) \propto -ln(posterior)
        potential = -(lnPrior + lnL)
        return potential

    def potential_fn_numpyro(self, autocorrelation_fn_data):
        """
        Wrapper for potential function to be used with numpyro. This allows one to call numpyro HMC with
        given autocorrelation_fn since the NumPyro HMC sampler potential function API only allows one to pass
        a function that takes a single argument.

        Args:
            autocorrelation_fn_data (array): autocorrelation function that the inference is being done for.

        Returns:
            potentical_function (function):
                The potential function to be used with NumPyro's HMC sampler.

        """
        return partial(self.potential_fn, autocorrelation_fn_data=autocorrelation_fn_data)

    # Quick function that does a quick fit to use as init values for the HMC
    def fit_one(self, autocorrelation_fn_data):
        """
        Fit a single quasar spectrum using the Adam optimizer

        Args:
            autocorrelation_fn_data (ndarray): shape (nvelocitybins,)
                data of the autocorrelation function we are doing the inference for.

        Returns:
            x_out (ndarray): shape (ndim,)
                best fit dimensionless parameter vector
            theta_out (ndarray): shape (ndim,)
                best fit parameter vector in physical units
            s_DR_out (ndarray): shape (nspec,)
                best fit data reduced intrinsic quasar spectrum
            losses (ndarray): shape (opt_nsteps,)
                loss function values at each iteration step
        """

        # Initialize the parameters for the new fit, just use the prior it is safer
        x = self.theta_to_x(self.theta_astro_inits)
        optimizer = optax.adam(self.opt_lr)
        opt_state = optimizer.init(x)
        losses = np.zeros(self.opt_nsteps)
        # Optimization loop for fitting input flux
        iterator = trange(self.opt_nsteps, leave=False)
        best_loss = np.inf  # Models are only saved if they reduce the validation loss
        for i in iterator:
            losses[i], grads = jax.value_and_grad(self.potential_fn, argnums=0)(x, autocorrelation_fn_data)
            if losses[i] < best_loss:
                x_out = x.copy()
                theta_out = self.x_to_theta(x_out)
                best_loss = losses[i]
            updates, opt_state = optimizer.update(grads, opt_state)
            x = optax.apply_updates(x, updates)

        return x_out, theta_out, losses

    # Now the HMC function
    def mcmc_one(self, key, x_opt, autocorrelation_fn_data):
        """
        HMC routine for a single quasar

        Args:
            key (JAX PRNG key)
                pseudo-random number generator key
            x_opt (ndarray): shape (ndim,)
                best fit dimensionless parameter vector
            autocorrelation_fn_data (ndarray): shape (22,)
                autocorrelation function that the inference is being done for.

        Returns:
            x_samples (ndarray):
                dimensionless samples saved for convenience; shape (mcmc_nsteps_tot, ndim)
            theta_samples (ndarray):
                parameter samples in physical units; shape (mcmc_nsteps_tot, ndim)
            lnP (ndarray):
                potential energy of HMC; shape (mcmc_nsteps_tot, )
            neff (ndarray):
                effective number of steps for each param; shape (mcmc_nsteps_tot, )
            neff_mean (float):
                average n_eff, here for convenience
            sec_per_neff (float):
                seconds per neff
            ms_per_step (float):
                milliseconds per step
            r_hat (ndarray):
                ratio of within-chain variance and posterior variance for convergence diagnostics;
                shape (mcmc_nsteps_tot, )
            r_hat_mean (float):
                average r_hat, here for convenience
            hmc_num_steps (int):
                number of steps in the Hamiltonian trajectory (for diagnostics)
            hmc_tree_depth (int):
                tree depth of the Hamiltonian trajectory (for diagnostics)
            total_time (float):
                total time for the MCMC run
        """

        # Initialize the MCMC parameters for each chain
        x_init = self.mcmc_init_x(self.mcmc_num_chains, self.mcmc_init_perturb, x_opt)
        # Instantiate the NUTS kernel and the mcmc object
        # Original line
        nuts_kernel = NUTS(potential_fn=self.potential_fn_numpyro(autocorrelation_fn_data),
                           adapt_step_size=True, dense_mass=self.mcmc_dense_mass,
                           max_tree_depth=self.mcmc_max_tree_depth)  # see how it changes by adapting mass matrix, maybe
        mcmc = MCMC(nuts_kernel, num_warmup=self.mcmc_warmup, num_samples=self.mcmc_nsteps,
                    num_chains=self.mcmc_num_chains,
                    chain_method='vectorized', jit_model_args=True)  # chain_method='sequential'
        # Run the MCMC
        start_time = time.time()
        mcmc.run(key, init_params=x_init, extra_fields=('potential_energy', 'num_steps'))
        total_time = time.time() - start_time
        print(f'Run time: {total_time:.2f} seconds')

        # Compute the neff and summarize cost
        az_summary = az.summary(az.from_numpyro(mcmc))
        neff = az_summary["ess_bulk"].to_numpy()
        neff_mean = np.mean(neff)
        r_hat = az_summary["r_hat"].to_numpy()
        r_hat_mean = np.mean(r_hat)
        sec_per_neff = (total_time / neff_mean)
        # Grab the samples and lnP
        x_samples = mcmc.get_samples(group_by_chain=True)
        theta_samples = self.x_to_theta(mcmc.get_samples())
        # lnP = -mcmc.get_extra_fields()['potential_energy']  # negative the potential is the log posterior ORIGINAL
        lnP = -mcmc.get_extra_fields(group_by_chain=True)['potential_energy']
        hmc_num_steps = mcmc.get_extra_fields()['num_steps']  # No of steps in Hamiltonian trajectory (for diagnostics).
        hmc_tree_depth = np.log2(hmc_num_steps).astype(int) + 1  # Tree depth of Hamiltonian traj (for diagnostics).
        hit_max_tree_depth = np.sum(hmc_tree_depth == self.mcmc_max_tree_depth)  # No of transitions that hit the max tree depth.
        ms_per_step = 1e3 * total_time / np.sum(hmc_num_steps)

        print(f"\n*** SUMMARY FOR HMC ***")
        print(f"total_time = {total_time} seconds for the HMC")
        print(f"total_steps = {np.sum(hmc_num_steps)} total steps")
        print(f"ms_per_step = {ms_per_step} ms per step of the HMC")
        print(f"n_eff_mean = {neff_mean} effective sample size.")  # with ntot = {self.mcmc_nsteps_tot} total samples.")
        print(f"sec_per_neff = {sec_per_neff:.3f} seconds per effective sample")
        print(f"r_hat_mean = {r_hat_mean}")
        print(f"max_tree_depth encountered = {hmc_tree_depth.max()} in chain")
        print(f"There were {hit_max_tree_depth} transitions that exceeded the max_tree_depth = {self.mcmc_max_tree_depth}")
        print("*************************\n")

        # Return the values needed
        return x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
               hmc_num_steps, hmc_tree_depth, total_time

    def mcmc(self, mock_datasets_to_fit, theta_true, theta_true_ngp, theta_true_og, gaussian_mocks,
             out_prefix=f'data/multiple_inferences/', debug=False):
        """Run the MCMC sampler.

        Args:
            mock_datasets_to_fit (ndarray): autocorrelation function data; shape (n_mock_datasets, velocity bins).
            theta_true (ndarray): true values of the parameters used to generate the mock datasets; shape
                (n_mock_datasets, 2). The first column is the mfp and the second is the mean flux.
            theta_true_ngp (ndarray): true values of the parameters used to generate the mock datasets, ngp
                interpolated to work with the covariance emulator. THIS IS TEMPORARY!!!
            theta_true_og (ndarray): true values of the parameters used to generate the mock datasets, non ngp
                interpolated to test not exactly at the boundaries. THIS IS TEMPORARY!!!
            gaussian_mocks (bool):
                Used to save the correct true model for each inference, if theta_true is provided.
            out_prefix (str): Prefix for the output files. Default is 'data/multiple_inferences/'.
            debug (bool): Show some plots that can be used as diagnostics.

        """
        # Now we set some class attributes used to save the results
        self.n_mock_datasets = mock_datasets_to_fit.shape[0]
        self.theta_true = jnp.atleast_2d(theta_true) if theta_true is not None else None
        self.theta_true_ngp = jnp.atleast_2d(theta_true_ngp) if theta_true_ngp is not None else None
        self.theta_true_og = jnp.atleast_2d(theta_true_og) if theta_true_og is not None else None
        self.autocorrelation_fn_mock_datasets = mock_datasets_to_fit
        self.x_true = jnp.atleast_2d(self.theta_to_x(theta_true)) if theta_true is not None else None
        self.x_true_og = jnp.atleast_2d(self.theta_to_x(theta_true_og)) if theta_true_og is not None else None  # THIS IS TEMPORARY!!!
        self.gaussian_mocks = gaussian_mocks

        self.mcmc_nsteps_tot = self.mcmc_nsteps * self.mcmc_num_chains

        self.neff = np.zeros((self.n_mock_datasets, self.ndim))  # Effective number of steps for each param
        self.neff_mean = np.zeros(self.n_mock_datasets)  # Average n_eff, here for convenience
        # Ratio of within-chain variance and posterior variance for convergence diagnostics
        self.r_hat = np.zeros((self.n_mock_datasets, self.ndim))
        self.r_hat_mean = np.zeros(self.n_mock_datasets)  # Average r_hat, here for convenience
        self.hmc_num_steps = np.zeros((self.n_mock_datasets,
                                       self.mcmc_nsteps_tot))  # Number of NUTS steps (lnP evals) for each real step
        self.hmc_tree_depth = np.zeros((self.n_mock_datasets,
                                        self.mcmc_nsteps_tot))  # Tree depth of the HMC trajectory (for diagnostics).
        self.sec_per_neff = np.zeros(self.n_mock_datasets)  # Seconds per neff
        self.ms_per_step = np.zeros(self.n_mock_datasets)  # Milliseconds per step
        self.runtime = np.zeros(self.n_mock_datasets)  # Runtime
        self.samples = np.zeros((self.n_mock_datasets, self.mcmc_nsteps_tot, self.ndim))  # Parameter samples
        self.x_samples = np.zeros((self.n_mock_datasets,
                                   self.mcmc_nsteps_tot,
                                   self.ndim))  # Dimensionless samples saved for convenience

        # self.ln_probs_x = np.zeros((self.n_mock_datasets, self.mcmc_nsteps_tot))  # ln_probs  ORIGINAL
        self.ln_probs_x = np.zeros((self.n_mock_datasets, self.mcmc_num_chains, self.mcmc_nsteps))

        # Some other things we need if truths were provided
        self.ln_prob_x_true = np.zeros(self.n_mock_datasets)  # ln_prob_x at true model x
        self.ln_like_true = np.zeros(self.n_mock_datasets)  # ln_like at true model
        self.ln_prior_x_true = np.zeros(self.n_mock_datasets)  # ln_prior_x at true model

        # Some other things we need if truths were provided that are temporary
        self.ln_prob_x_true_og = np.zeros(self.n_mock_datasets)  # ln_prob_x at true model x

        # Loop over the entire training set and fit every object
        iterator = trange(self.n_mock_datasets, leave=True)

        for imock in iterator:
            iterator.set_description('mcmc> MCMC for mock dataset no: {:d}'.format(imock))

            # Perform the optimization to get the starting point
            autocorrelation_fn_data = mock_datasets_to_fit[imock, :]
            x_opt, theta_opt, losses = self.fit_one(autocorrelation_fn_data)

            # If the true theta is passed in evaluate lnP_x_true, lnlike_true, ln_prior_x_true
            if self.x_true is not None:
                # First, we need to evaluate the ln_prob_x_true, ln_like_true, ln_prior_x_true
                # ln_prob_x_true is the log probability of the true model x, which means it is the
                # negative of the potential function evaluated at the true model x
                self.ln_prob_x_true[imock] = -self.potential_fn(x_astro=self.x_true[imock, :],
                                                                autocorrelation_fn_data=autocorrelation_fn_data)

                self.ln_like_true[imock] = self.ln_gaussian_likelihood(x_astro=self.x_true[imock, :],
                                                                       autocorrelation_fn_data=autocorrelation_fn_data)

                self.ln_prior_x_true[imock] = self.ln_prior(x_astro=self.x_true[imock, :])

                self.ln_prob_x_true_og[imock] = -self.potential_fn(x_astro=self.x_true_og[imock, :],
                                                                   autocorrelation_fn_data=autocorrelation_fn_data)

            # Split the key
            self.key, subkey = random.split(self.key)

            # do the mcmc for this mock dataset
            imock_mcmc_results = self.mcmc_one(key=subkey,
                                               x_opt=x_opt,
                                               autocorrelation_fn_data=autocorrelation_fn_data)

            # unpack the results
            x_samples = imock_mcmc_results[0]
            self.x_samples[imock, ...] = np.reshape(x_samples,
                                                    (self.mcmc_num_chains * self.mcmc_nsteps, self.ndim))
            self.samples[imock, ...] = imock_mcmc_results[1]
            self.ln_probs_x[imock, :] = imock_mcmc_results[2]
            self.neff[imock, :] = imock_mcmc_results[3]
            self.neff_mean[imock] = imock_mcmc_results[4]
            self.sec_per_neff[imock] = imock_mcmc_results[5]
            self.ms_per_step[imock] = imock_mcmc_results[6]
            self.r_hat[imock, :] = imock_mcmc_results[7]
            self.r_hat_mean[imock] = imock_mcmc_results[8]
            self.hmc_num_steps[imock, :] = imock_mcmc_results[9]
            self.hmc_tree_depth[imock, :] = imock_mcmc_results[10]
            self.runtime[imock] = imock_mcmc_results[11]

            # Make some diagnostic plots
            if out_prefix is not None or debug:
                indx = imock
                qa_imock = 'imock_{:03d}'.format(indx)
                walkerfile = out_prefix + '_walkers_' + qa_imock + '.pdf'
                cornerfile = out_prefix + '_corner_' + qa_imock + '.pdf'
                x_cornerfile = out_prefix + '_x-corner_' + qa_imock + '.pdf'
                infer_file = out_prefix + '_infer_' + qa_imock + '.pdf'

                _x_true = self.x_true[imock, :] if self.x_true is not None else None
                _theta_true = self.theta_true[imock, :] if self.theta_true is not None else None
                # embed(header='mcmc> debug here')
                _ln_prob_x_true = self.ln_prob_x_true[imock] if self.theta_true is not None else None
                # _ln_prob_true = self.ln_prob_true[imock, :] if self.theta_true is not None else None

                # Produce the plots, here is the label we use for this task
                label = [r'$\lambda_{\rm mfp}$', r'$\langle F \rangle$']
                # Walker plot
                walker_plot(np.swapaxes(x_samples, 0, 1), label,
                            truths=self.x_true[imock, :] if self.x_true is not None else None,
                            walkerfile=walkerfile, linewidth=1.0)
                plt.close()
                # Corner plot in HMC units
                corner_plot(self.x_samples[imock, ...], label,
                            theta_true=_x_true if self.x_true is not None else None,
                            cornerfile=x_cornerfile)
                plt.close()
                # Corner plot in physical units
                corner_plot(self.samples[imock, ...], label,
                            theta_true=self.theta_true[imock, :] if self.theta_true is not None else None,
                            cornerfile=cornerfile)
                plt.close()
                # Inferred model plot
                self.inferred_model_plot(theta_samples=self.samples[imock, ...],
                                         theta_true=self.theta_true[imock, :] if self.theta_true is not None else None,
                                         autocorrelation_fn_data=autocorrelation_fn_data,
                                         n_posterior_draws=250,
                                         infer_file=infer_file)
                plt.close()

        plt.close('all')

        if out_prefix is not None:
            mcmc_savefile = out_prefix + '.hdf5'
            self.mcmc_save(mcmc_savefile)

    def mcmc_save(self, mcmc_savefile):
        """
        Save the MCMC results to an HDF5 file

        Args:
            mcmc_savefile (str):
                output file for MCMC results

        Returns:

        """

        with h5py.File(mcmc_savefile, 'w') as f:
            group = f.create_group('mcmc')
            # Set the attribute parameters of the MCMC
            group.attrs['mcmc_nsteps'] = self.mcmc_nsteps
            group.attrs['mcmc_num_chains'] = self.mcmc_num_chains
            group.attrs['mcmc_warmup'] = self.mcmc_warmup
            group.attrs['mcmc_dense_mass'] = self.mcmc_dense_mass
            group.attrs['mcmc_max_tree_depth'] = self.mcmc_max_tree_depth
            group.attrs['mcmc_init_perturb'] = self.mcmc_init_perturb
            group.attrs['mcmc_nsteps_tot'] = self.mcmc_nsteps_tot
            # Some other parameters of the run
            group.attrs['n_mock_datasets'] = self.n_mock_datasets
            group.attrs['ndim'] = self.ndim
            # MCMC results
            group.create_dataset('neff', data=self.neff)
            group.create_dataset('neff_mean', data=self.neff_mean)
            group.create_dataset('sec_per_neff', data=self.sec_per_neff)
            group.create_dataset('ms_per_step', data=self.ms_per_step)
            group.create_dataset('r_hat', data=self.r_hat)
            group.create_dataset('r_hat_mean', data=self.r_hat_mean)
            group.create_dataset('hmc_num_steps', data=self.hmc_num_steps)
            group.create_dataset('hmc_tree_depth', data=self.hmc_tree_depth)
            group.create_dataset('runtime', data=self.runtime)
            group.create_dataset('samples', data=self.samples)
            group.create_dataset('x_samples', data=self.x_samples)
            group.create_dataset('ln_probs_x', data=self.ln_probs_x)
            # Save the data
            group.create_dataset('autocorrelation_fn_mock_datasets', data=self.autocorrelation_fn_mock_datasets)
            # Some other things related to truths
            if self.theta_true is not None:
                group.create_dataset('ln_prob_x_true', data=self.ln_prob_x_true)
                group.create_dataset('ln_like_true', data=self.ln_like_true)
                group.create_dataset('ln_prior_x_true', data=self.ln_prior_x_true)
                group.create_dataset('theta_true', data=self.theta_true)
                group.create_dataset('x_true', data=self.x_true)
                # temporary
                group.create_dataset('ln_prob_x_true_og', data=self.ln_prob_x_true_og)

    # functions to do the MCMC initialization
    def x_minmax(self):
        x_min, x_max = bounded_theta_to_x(self.theta_astro_mins, self.theta_astro_ranges), \
                        bounded_theta_to_x(self.theta_astro_maxs, self.theta_astro_ranges)

        return x_min, x_max

    def mcmc_init_x(self, nwalkers, perturb, x_opt):
        """

        Args:
            nwalkers:
            perturb:
            x_opt:

        Returns:

        """

        x_min, x_max = self.x_minmax()
        delta_x = x_max - x_min

        self.key, subkey = random.split(self.key)
        deviates = perturb * random.normal(subkey, (nwalkers, self.ndim))

        theta_init = jnp.array([[jnp.clip(x_opt[i] + delta_x[i] * deviates[j, i],
                                          x_min[i], x_max[i]) for i in range(self.ndim)]
                                for j in range(nwalkers)])

        return theta_init.squeeze()

    def explore_logP(self, autocorrelation_fn_data):
        """
        Explore the negative of the Potential function (prop to logL + logPrior by plotting it as a
        function of the parameters).
        Args:
            autocorrelation_fn_data (ndarray):
                autocorrelation function data; shape (n_data, )
        """
        # create a grid of for the theta parameters
        mfp_grid = np.linspace(self.theta_astro_ranges[0][0], self.theta_astro_ranges[0][1], 100)
        mean_flux_grid = np.linspace(self.theta_astro_ranges[1][0], self.theta_astro_ranges[1][1], 90)

        # create the empty array with the likelihood values
        logP_grid = np.zeros((len(mfp_grid), len(mean_flux_grid)))

        # loop over the grid and compute the likelihood
        for i, mfp in enumerate(mfp_grid):
            for j, mean_flux in enumerate(mean_flux_grid):
                logP_grid[i, j] = -self.potential_fn(self.theta_to_x(np.array([mfp, mean_flux])),
                                                     autocorrelation_fn_data)

        return mfp_grid, mean_flux_grid, logP_grid

    def inferred_model_plot(self, theta_samples, autocorrelation_fn_data, n_posterior_draws, infer_file,
                            theta_true=None):
        """
        Plot the inferred model with the data
        Args:
            theta_samples (ndarray):
                MCMC samples; shape (n_samples, n_dim)
            autocorrelation_fn_data (ndarray):
                autocorrelation function data; shape (n_data, )
            n_posterior_draws (int):
                number of posterior draws to plot
            infer_file (str):
                output file name
            theta_true (ndarray):
                true parameter values; shape (n_dim, )
        """
        # calculate the inferred theta values
        inferred_theta = np.median(theta_samples, axis=0)
        # calculate the inferred model, first emulate all the samples
        sampled_models = self.laf_mean_emulator.predict(theta_samples)
        inferred_model = np.median(sampled_models, axis=0)
        # randomly select the sampled models to plot
        self.key, subkey = random.split(self.key)
        sampled_models_to_plot = random.permutation(key=subkey,
                                                    x=sampled_models,
                                                    axis=0,
                                                    independent=False)[:n_posterior_draws]

        # THIS IS VERY PROBLEM SPECIFIC, TO GENERALIZE THIS PART, THE ERROR ON THE OBSERVATION SHOULD BE A PARAMETER
        # OF THIS METHOD
        # get the errors of the mock dataset, defined as the diagonal elements of the covariance matrix of the model
        # that is nearest to the inferred model
        _1, vbins, mfp_grid, mean_flux_grid = self.dataset_loader.get_attributes(redshift_idx=self.z_ti)

        ngp_mfp = NGP1D(np.array([inferred_theta[0]]), mfp_grid)
        ngp_mean_flux = NGP1D(np.array([inferred_theta[1]]), mean_flux_grid)

        mock_error_cov = self.dataset_loader.get_model(redshift_idx=self.z_ti,
                                                       mfp_idx=ngp_mfp[0],
                                                       mean_flux_idx=ngp_mean_flux[0])[-1]
        mock_error = np.sqrt(np.diag(mock_error_cov))

        # make the plot
        with plt.style.context(['science', 'no-latex']):
            fig, ax = plt.subplots(1, 1, figsize=(9, 4))

            # the inferred model
            ax.plot(vbins, inferred_model, label='Inferred Model', c='red', lw=2)

            # randomly sampled models
            for i in range(n_posterior_draws):
                ax.plot(vbins, sampled_models_to_plot[i], alpha=0.05, lw=1, c='b')
            ax.plot([], c='b', alpha=0.4, lw=1, label='Posterior Draws')

            # the data with error bars
            ax.errorbar(vbins, autocorrelation_fn_data, yerr=mock_error, marker='o', label='Mock Data', c='k', ls='None')

            if theta_true is not None:

                # plot the true model
                if self.gaussian_mocks:
                    # if doing gaussian mocks with the emulator, use the following:
                    true_model_autocorrelation = self.laf_mean_emulator.predict(theta_true).ravel()
                else:
                    # make sure that theta true is on th grid for this, THIS IS TEMPORARY!!!
                    # TO MAKE THIS MORE GENERAL, THE TRUE MODEL SHOULD BE A PARAMETER OF THIS METHOD
                    true_mfp_ngp = NGP1D(np.array([theta_true[0]]), mfp_grid)
                    true_mean_flux_ngp = NGP1D(np.array([theta_true[1]]), mean_flux_grid)

                    true_model_all = self.dataset_loader.get_model(redshift_idx=self.z_ti,
                                                                   mfp_idx=true_mfp_ngp[0],
                                                                   mean_flux_idx=true_mean_flux_ngp[0])

                    # if doing actual mocks, use this one instead:
                    true_model_autocorrelation = true_model_all[-2]

                ax.plot(vbins, true_model_autocorrelation, '--', label='True Model', c='green')

            ax.legend()
            ax.set_xlabel(r'$v$ [km/s]', fontsize=14)
            ax.set_ylabel(r'Correlation', fontsize=14)

            plt.savefig(infer_file, bbox_inches='tight')


# A class to do the inference
class HMCInferenceConvexHullPrior:
    """
    A class to do the inference using HMC sampling for the LAF, with only the mfp and mean flux as astrophysical
    parameters.

    """

    def __init__(self, ndim, convex_hull, mean_emulator, cov_emulator, inferred_plots_dict,
                 opt_nsteps=150, opt_lr=0.01, mcmc_nsteps=1000, mcmc_num_chains=4, mcmc_warmup=1000,
                 mcmc_init_perturb=0.05,  mcmc_max_tree_depth=10, mcmc_dense_mass=True, key=random.PRNGKey(42)):
        """
        Initialize the HMCInference class.

        Args:
            ndim (int):
                Number of dimensions of the parameter space.
            convex_hull (ConvexHull):
                A scipy.spatial.ConvexHull object that defines the convex hull prior.
            mean_emulator (TrainerModule):
                An emulator for the mean statistic.
            cov_emulator (LAFDatasetLoader):
                An emulator for the covariance matrices.
            inferred_plots_dict (dict):
                A dictionary containing the paths to save the plots of the inferred model. This should contain the
                following keys:
                'params_labels' (list of str): labels for the parameters
                'ylim' (list of float): limits for the y-axis for the inferred model plot
                'xlabel' (str): label for the x-axis for the inferred model plot
                'ylabel' (str): label for the y-axis for the inferred model plot
                'xvalues' (list of float): values for the x-axis for the inferred model plot
            opt_nsteps (int):
                Number of the quick optimization steps used for initializing x for the HMC.
            opt_lr (float):
                Learning rate for the quick optimization steps used for initializing x for the HMC.
            mcmc_nsteps (int):
                number of steps for MCMC per chain. Total steps is this number times mcmc_num_chains
            mcmc_num_chains (int):
                number of MCMC chains
            mcmc_warmup (int):
                number of warmup steps for MCMC per chain
            mcmc_init_perturb (float):
                fractional amount to perturb about optimum for MCMC chains
            mcmc_max_tree_depth (int):
                max depth of the binary tree created during the doubling scheme of NUTS HMC sampler
            mcmc_dense_mass (bool):
                flag indicating if mass matrix is dense or diagonal
            key (JAX PRNG key):
                pseudo-random number generator key.
        """
        # Store the parameters
        self.ndim = ndim  # this is hardcoded for our case
        self.convex_hull = convex_hull
        self.hull_equations = jnp.array(convex_hull.equations)
        self.theta_astro_ranges = [(self.convex_hull.min_bound[i], self.convex_hull.max_bound[i]) for i in range(ndim)]
        self.theta_astro_inits = tuple([np.mean([tup[0], tup[1]]) for tup in self.theta_astro_ranges])  # mean of the ranges
        self.theta_astro_mins = jnp.array([astro_par_range[0] for astro_par_range in self.theta_astro_ranges])
        self.theta_astro_maxs = jnp.array([astro_par_range[1] for astro_par_range in self.theta_astro_ranges])

        # Set astro parameter priors (Smoothed Box Prior is a differentiable uniform distribution)
        self.x_astro_priors = [bounded_variable_lnP, bounded_variable_lnP]

        # Store the emulators
        self.mean_emulator = mean_emulator
        self.cov_emulator = cov_emulator

        # Store the inferred plots dictionary
        self.inferred_plots_dict = inferred_plots_dict

        # Set the optimizer parameters
        self.opt_nsteps = opt_nsteps
        self.opt_lr = opt_lr

        # Set the MCMC parameters
        self.mcmc_nsteps = mcmc_nsteps  # Number of steps
        self.mcmc_num_chains = mcmc_num_chains  # Number of walkers
        self.mcmc_warmup = mcmc_warmup  # Number of burning steps
        self.mcmc_max_tree_depth = mcmc_max_tree_depth
        self.mcmc_dense_mass = mcmc_dense_mass
        self.mcmc_init_perturb = mcmc_init_perturb  # fractional amount to perturb about optimum for MCMC chains
        self.key = key  # random number generator key

        # These are some class attributes used in the case of multiple inferences that are used to save the results
        self.n_mock_datasets = None
        self.theta_true = None
        self.theta_true_ngp = None  # THIS IS TEMPORARY
        self.theta_true_og = None  # THIS IS TEMPORARY
        self.gaussian_mocks = None  # this is used to save the correct true model, which depends on the type of mocks
        self.autocorrelation_fn_mock_datasets = None
        self.x_true = None
        self.mcmc_nsteps_tot = None
        self.neff = None
        self.neff_mean = None
        self.r_hat = None
        self.r_hat_mean = None
        self.hmc_num_steps = None
        self.hmc_tree_depth = None
        self.sec_per_neff = None
        self.ms_per_step = None
        self.runtime = None
        self.samples = None
        self.x_samples = None
        self.ln_probs_x = None
        # Some other things we need if truths were provided
        self.ln_prob_x_true = None
        self.ln_like_true = None
        self.ln_prior_x_true = None

    # functions to transform between theta and x
    # x to theta
    @partial(jit, static_argnums=(0,))
    def x_to_theta(self, x):
        """
        Transform from HMC parameter vector x into parameter vector theta. This is the vectorized version of the
        function _x_to_theta below.

        Args:
            x (jax.numpy array):
               HMC parameter vector. Shape (nsamples, ndim) or (ndim,)
        Returns:
            theta (jax.numpy array):
               Parameter vector. Shape (nsamples, ndim) or (ndim,)
        """
        theta = jax.vmap(self._x_to_theta, in_axes=0, out_axes=0)(jnp.atleast_2d(x))
        return theta.squeeze()

    @partial(jit, static_argnums=(0,))
    def _x_to_theta(self, x):
        """
        Transform from HMC parameter vector x into parameter vector theta. This is the single element version called by
         the vectorized version above.

        Args:
            x (jax.numpy array):
               HMC parameter vector. Shape (ndim,)
        Returns:
            theta (jax.numpy array):
               Parameter vector. Shape (ndim,)
        """

        theta = x_to_bounded_theta(x, self.theta_astro_ranges)
        return theta

    # theta to x
    @partial(jit, static_argnums=(0,))
    def theta_to_x(self, theta):
        """
         Transform from parameter vector theta into the HMC parameter vector theta. This is the vectorized version of
         the function _theta_to_x below.

         Args:
             theta (jax.numpy array):
                Parameter vector. Shape (nsamples, ndim) or (ndim,)

         Returns:
             x (jax.numpy array):
                HMC parameter vector. Shape (nsamples, ndim) or (ndim,)
         """
        x = jax.vmap(self._theta_to_x, in_axes=0, out_axes=0)(jnp.atleast_2d(theta))
        return x.squeeze()

    @partial(jit, static_argnums=(0,))
    def _theta_to_x(self, theta):
        """
        Transform from parameter vector x into parameter vector theta. This is the single element version called by
        the vectorized version above.

        Args:
             theta (jax.numpy array):
                Parameter vector. Shape (nsamples, ndim) or (ndim,)

        Returns:
             x (jax.numpy array):
                HMC parameter vector. Shape (nsamples, ndim) or (ndim,)
        """
        x = bounded_theta_to_x(theta, self.theta_astro_ranges)
        return x

    # method to check if a point is inside the convex hull using JAX
    @partial(jit, static_argnums=(0,))
    def point_in_hull(self, theta_astro, tolerance=1e-12):
        """
        Verifies if a point is inside a convex hull. Taken from:
        https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
        The text says:
        In words, a point is in the hull if and only if for every equation (describing the facets) the dot product
        between the point and the normal vector (eq[:-1]) plus the offset (eq[-1]) is less than or equal to zero. You
        may want to compare to a small, positive constant tolerance = 1e-12 rather than to zero because of issues of
        numerical precision (otherwise, you may find that a vertex of the convex hull is not in the convex hull).

        This version uses jax numpy and jit to speed up the process.

        Args:
            theta_astro (ndarray): shape = (nastro,)
                astrophysical parameter vector

        Returns:
            prior (float):
                Prior on these model parameters
        """
        # Compute dot products and add offsets
        dot_products = jnp.dot(self.hull_equations[:, :-1], theta_astro) + self.hull_equations[:, -1]

        # Check if all dot products are within tolerance
        r = jnp.all(dot_products <= tolerance)

        return r

    # Now the log functions for the prior and likelihood evaluations
    @partial(jit, static_argnums=(0,))
    def ln_prior(self, x_astro):
        """
        Compute the prior on astro_params

        Args:
            x_astro (ndarray): shape = (nastro,)
                dimensionless astrophysical parameter vector

        Returns:
            prior (float):
                Prior on these model parameters
        """
        # the prior has two parts, first we need to evaluate it in the original theta units
        # evaluate if the parameters are within the prior boundaries defined by the convex hull (in theta units)
        condition = self.point_in_hull(self.x_to_theta(x_astro))

        # return the log prior, using a uniform distribution within the convex hull
        prior = 0.  # jnp.where(condition, 0.0, -jnp.inf)
        extra_terms = 0.

        # then we need to add the extra terms that come from the transformation of variables
        for x_ast in x_astro:
            extra_terms += bounded_variable_lnP(x_ast)

        prior = jnp.where(condition, prior + extra_terms, -jnp.inf)  # or should it be zero instead of -inf?

        return prior

    # function that uses the emulators to get the mean autocorrelation function and its corresponding covariance matrix
    @partial(jit, static_argnums=(0,))
    def get_mean_and_covar(self, theta_astro):
        """
        Returns the mean autocorrelation function and its corresponding covariance matrix given the astrophysical
        parameters.

        Args:
            theta_astro (array): theta_astro[0] is mfp, theta_astro[1] is mean_flux in physical units.

        Returns:
            (mean_statistic, covariance_matrix)
        """
        # get the mean statsitic from the neural emulator
        # mean_statistic = self.mean_emulator.predict(theta_astro).ravel()  # Is ravel is needed? ORIGINAL
        mean_statistic = self.mean_emulator.predict_unlogged(theta_astro).ravel()  # Is ravel is needed?
        # get the covariance matrix from the neural emulator
        covariance_matrix = self.cov_emulator.predict_single(theta_astro)

        return mean_statistic, covariance_matrix

    # Gaussian likelihood
    @partial(jit, static_argnums=(0,))
    def ln_gaussian_likelihood(self, x_astro, observed_data):
        """
        Natural logarithm of the gaussian likelihood.

        Args:
            x_astro (jnp.array): model parameters in dimensionless units.
            observed_data (jnp.array): observed data that the inference is being done for.

        Returns:
            float: ln of gaussian likelihood.
        """
        # convert to physical units
        theta_astro = self.x_to_theta(x_astro)
        # obtain the mean autocorrelation model and its corresponding covariance matrix
        mean_statistic, covariance_matrix = self.get_mean_and_covar(theta_astro=theta_astro)
        # compute and return the gaussian likelihood
        return jax.scipy.stats.multivariate_normal.logpdf(x=observed_data,
                                                          mean=mean_statistic,
                                                          cov=covariance_matrix)

    # Now the potential function and its helper used for the numpyro NUTS sampler
    @partial(jit, static_argnums=(0,))
    def potential_fn(self, x_astro, observed_data):
        """
            Potential function for the MCMC.

            Args:
                x_astro (array): model parameters in dimensionless units.
                observed_data (array): observed data that the inference is being done for.

            Returns:
                float: potential.
            """
        # ln(prior) calculated
        lnPrior = self.ln_prior(x_astro)
        # ln(likelihood) calculated
        lnL = self.ln_gaussian_likelihood(x_astro, observed_data)
        # Calculate the potential = -(ln(likelihood) + ln(prior)) \propto -ln(posterior)
        potential = -(lnPrior + lnL)
        return potential

    # A function that computes lnP in the theta units
    @partial(jit, static_argnums=(0,))
    def lnP_theta(self, x_astro, observed_data):
        """
        Computes the log of the posterior probability density function in the theta units, up to a constant.

        Args:
            x_astro (array): model parameters in dimensionless units.
            observed_data (array): observed data that the inference is being done for.

        Returns:
            float: ln of posterior probability density function in theta units.
        """
        # the prior has only one part: we need to evaluate it in the original theta units
        # evaluate if the parameters are within the prior boundaries defined by the convex hull (in theta units)
        condition = self.point_in_hull(self.x_to_theta(x_astro))

        # return the log prior, using a uniform distribution within the convex hull
        ln_Prior = jnp.where(condition, 0., -jnp.inf)  # or should it be zero instead of -inf?
        # ln(likelihood) calculated
        ln_Likelihood = self.ln_gaussian_likelihood(x_astro, observed_data)
        # Calculate the potential = -(ln(likelihood) + ln(prior)) \propto -ln(posterior)
        lnP_theta = ln_Prior + ln_Likelihood
        return lnP_theta

    # A function that computes lnP in the theta units
    @partial(jit, static_argnums=(0,))
    def extra_term(self, x_astro):
        """
        Computes the extra term that comes from the transformation of variables.

        Args:
            x_astro (array): model parameters in dimensionless units.

        Returns:
            extra_terms (float): extra term that comes from the transformation of variables.
        """
        # # the prior has two parts, first we need to evaluate it in the original theta units
        # # evaluate if the parameters are within the prior boundaries defined by the convex hull (in theta units)
        # condition = self.point_in_hull(self.x_to_theta(x_astro))

        # return the log prior, using a uniform distribution within the convex hull
        extra_terms = 0.

        # then we need to add the extra terms that come from the transformation of variables
        for x_ast in x_astro:
            extra_terms += bounded_variable_lnP(x_ast)

        # # the extra term is only valid if the parameters are within the prior boundaries defined by the convex hull
        # extra_terms = jnp.where(condition, extra_terms, -jnp.inf)  # or should it be zero instead of -inf?
        return extra_terms

    def potential_fn_numpyro(self, observed_data):
        """
        Wrapper for potential function to be used with numpyro. This allows one to call numpyro HMC with a
        given observed data array, since the NumPyro HMC sampler potential function API only allows one to pass
        a function that takes a single argument.

        Args:
            observed_data (array): observed data that the inference is being done for.

        Returns:
            potentical_function (function):
                The potential function to be used with NumPyro's HMC sampler.

        """
        return partial(self.potential_fn, observed_data=observed_data)

    # Quick function that does a quick fit to use as init values for the HMC
    def fit_one(self, observed_data):
        """
        Fit a single observed data array to get initial values for the HMC sampler.

        Args:
            observed_data (ndarray): shape (nbins,)
                data of the autocorrelation function we are doing the inference for.

        Returns:
            x_out (ndarray): shape (ndim,)
                best fit dimensionless parameter vector
            theta_out (ndarray): shape (ndim,)
                best fit parameter vector in physical units
            losses (ndarray): shape (opt_nsteps,)
                loss function values at each iteration step
        """

        # Initialize the parameters for the new fit, just use the prior it is safer
        x = self.theta_to_x(self.theta_astro_inits)
        optimizer = optax.adam(self.opt_lr)
        opt_state = optimizer.init(x)
        losses = np.zeros(self.opt_nsteps)
        # Optimization loop for fitting the observed data using the emulators
        iterator = trange(self.opt_nsteps, leave=False)
        best_loss = np.inf  # Models are only saved if they reduce the validation loss
        for i in iterator:
            losses[i], grads = jax.value_and_grad(self.potential_fn, argnums=0)(x, observed_data)
            if losses[i] < best_loss:
                x_out = x.copy()
                theta_out = self.x_to_theta(x_out)
                best_loss = losses[i]
            updates, opt_state = optimizer.update(grads, opt_state)
            x = optax.apply_updates(x, updates)

        return x_out, theta_out, losses

    # Now the HMC function
    def mcmc_one(self, key, x_opt, observed_data):
        """
        HMC routine for a single observed data array.

        Args:
            key (JAX PRNG key)
                pseudo-random number generator key
            x_opt (ndarray): shape (ndim,)
                best fit dimensionless parameter vector
            observed_data (ndarray): shape (nbins,)
                observed data that the inference is being done for.

        Returns:
            x_samples (ndarray):
                dimensionless samples saved for convenience; shape (mcmc_nsteps_tot, ndim)
            theta_samples (ndarray):
                parameter samples in physical units; shape (mcmc_nsteps_tot, ndim)
            lnP (ndarray):
                potential energy of HMC; shape (mcmc_nsteps_tot, )
            neff (ndarray):
                effective number of steps for each param; shape (mcmc_nsteps_tot, )
            neff_mean (float):
                average n_eff, here for convenience
            sec_per_neff (float):
                seconds per neff
            ms_per_step (float):
                milliseconds per step
            r_hat (ndarray):
                ratio of within-chain variance and posterior variance for convergence diagnostics;
                shape (mcmc_nsteps_tot, )
            r_hat_mean (float):
                average r_hat, here for convenience
            hmc_num_steps (int):
                number of steps in the Hamiltonian trajectory (for diagnostics)
            hmc_tree_depth (int):
                tree depth of the Hamiltonian trajectory (for diagnostics)
            total_time (float):
                total time for the MCMC run
        """

        # Initialize the MCMC parameters for each chain
        x_init = self.mcmc_init_x(self.mcmc_num_chains, self.mcmc_init_perturb, x_opt)
        # Instantiate the NUTS kernel and the mcmc object
        nuts_kernel = NUTS(potential_fn=self.potential_fn_numpyro(observed_data),
                           adapt_step_size=True, dense_mass=self.mcmc_dense_mass,
                           max_tree_depth=self.mcmc_max_tree_depth)
        mcmc = MCMC(nuts_kernel, num_warmup=self.mcmc_warmup, num_samples=self.mcmc_nsteps,
                    num_chains=self.mcmc_num_chains,
                    chain_method='vectorized', jit_model_args=True)  # chain_method='sequential'
        # Run the MCMC
        start_time = time.time()
        mcmc.run(key, init_params=x_init, extra_fields=('potential_energy', 'num_steps'))
        total_time = time.time() - start_time
        print(f'Run time: {total_time:.2f} seconds')

        # Compute the neff and summarize cost
        az_summary = az.summary(az.from_numpyro(mcmc))
        neff = az_summary["ess_bulk"].to_numpy()
        neff_mean = np.mean(neff)
        r_hat = az_summary["r_hat"].to_numpy()
        r_hat_mean = np.mean(r_hat)
        sec_per_neff = (total_time / neff_mean)
        # Grab the samples and lnP
        x_samples = mcmc.get_samples(group_by_chain=True)
        theta_samples = self.x_to_theta(mcmc.get_samples())
        # lnP = -mcmc.get_extra_fields()['potential_energy']  # negative the potential is the log posterior ORIGINAL
        lnP = -mcmc.get_extra_fields(group_by_chain=True)['potential_energy']
        hmc_num_steps = mcmc.get_extra_fields()['num_steps']  # No of steps in Hamiltonian trajectory (for diagnostics).
        hmc_tree_depth = np.log2(hmc_num_steps).astype(int) + 1  # Tree depth of Hamiltonian traj (for diagnostics).
        hit_max_tree_depth = np.sum(hmc_tree_depth == self.mcmc_max_tree_depth)  # No of transitions that hit the max tree depth.
        ms_per_step = 1e3 * total_time / np.sum(hmc_num_steps)

        print(f"\n*** SUMMARY FOR HMC ***")
        print(f"total_time = {total_time} seconds for the HMC")
        print(f"total_steps = {np.sum(hmc_num_steps)} total steps")
        print(f"ms_per_step = {ms_per_step} ms per step of the HMC")
        print(f"n_eff_mean = {neff_mean} effective sample size.")  # with ntot = {self.mcmc_nsteps_tot} total samples.")
        print(f"sec_per_neff = {sec_per_neff:.3f} seconds per effective sample")
        print(f"r_hat_mean = {r_hat_mean}")
        print(f"max_tree_depth encountered = {hmc_tree_depth.max()} in chain")
        print(f"There were {hit_max_tree_depth} transitions that exceeded the max_tree_depth = {self.mcmc_max_tree_depth}")
        print("*************************\n")

        # Return the values needed
        return x_samples, theta_samples, lnP, neff, neff_mean, sec_per_neff, ms_per_step, r_hat, r_hat_mean, \
               hmc_num_steps, hmc_tree_depth, total_time

    def mcmc(self, mock_datasets_to_fit, theta_true, theta_true_og, gaussian_mocks,
             out_prefix=f'data/multiple_inferences/', debug=False):
        """Run the MCMC sampler.

        Args:
            mock_datasets_to_fit (ndarray): autocorrelation function data; shape (n_mock_datasets, velocity bins).
            theta_true (ndarray): true values of the parameters used to generate the mock datasets; shape
                (n_mock_datasets, 2). The first column is the mfp and the second is the mean flux.
            theta_true_og (ndarray): true values of the parameters used to generate the mock datasets, non ngp
                interpolated to test not exactly at the boundaries. THIS IS TEMPORARY!!!
            gaussian_mocks (bool):
                Used to save the correct true model for each inference, if theta_true is provided.
            out_prefix (str): Prefix for the output files. Default is 'data/multiple_inferences/'.
            debug (bool): Show some plots that can be used as diagnostics.

        """
        # Now we set some class attributes used to save the results
        self.n_mock_datasets = mock_datasets_to_fit.shape[0]
        self.theta_true = jnp.atleast_2d(theta_true) if theta_true is not None else None
        # self.theta_true_ngp = jnp.atleast_2d(theta_true_ngp) if theta_true_ngp is not None else None
        self.theta_true_og = jnp.atleast_2d(theta_true_og) if theta_true_og is not None else None
        self.autocorrelation_fn_mock_datasets = mock_datasets_to_fit
        self.x_true = jnp.atleast_2d(self.theta_to_x(theta_true)) if theta_true is not None else None
        self.x_true_og = jnp.atleast_2d(self.theta_to_x(theta_true_og)) if theta_true_og is not None else None  # THIS IS TEMPORARY!!!
        self.gaussian_mocks = gaussian_mocks

        self.mcmc_nsteps_tot = self.mcmc_nsteps * self.mcmc_num_chains

        self.neff = np.zeros((self.n_mock_datasets, self.ndim))  # Effective number of steps for each param
        self.neff_mean = np.zeros(self.n_mock_datasets)  # Average n_eff, here for convenience
        # Ratio of within-chain variance and posterior variance for convergence diagnostics
        self.r_hat = np.zeros((self.n_mock_datasets, self.ndim))
        self.r_hat_mean = np.zeros(self.n_mock_datasets)  # Average r_hat, here for convenience
        self.hmc_num_steps = np.zeros((self.n_mock_datasets,
                                       self.mcmc_nsteps_tot))  # Number of NUTS steps (lnP evals) for each real step
        self.hmc_tree_depth = np.zeros((self.n_mock_datasets,
                                        self.mcmc_nsteps_tot))  # Tree depth of the HMC trajectory (for diagnostics).
        self.sec_per_neff = np.zeros(self.n_mock_datasets)  # Seconds per neff
        self.ms_per_step = np.zeros(self.n_mock_datasets)  # Milliseconds per step
        self.runtime = np.zeros(self.n_mock_datasets)  # Runtime
        self.samples = np.zeros((self.n_mock_datasets, self.mcmc_nsteps_tot, self.ndim))  # Parameter samples
        self.x_samples = np.zeros((self.n_mock_datasets,
                                   self.mcmc_nsteps_tot,
                                   self.ndim))  # Dimensionless samples saved for convenience

        # self.ln_probs_x = np.zeros((self.n_mock_datasets, self.mcmc_nsteps_tot))  # ln_probs  ORIGINAL
        self.ln_probs_x = np.zeros((self.n_mock_datasets, self.mcmc_num_chains, self.mcmc_nsteps))
        self.ln_probs_theta = np.zeros((self.n_mock_datasets, self.mcmc_num_chains, self.mcmc_nsteps))  # ln_probs

        # Some other things we need if truths were provided
        self.ln_prob_x_true = np.zeros(self.n_mock_datasets)  # ln_prob_x at true model x
        self.ln_like_true = np.zeros(self.n_mock_datasets)  # ln_like at true model
        self.ln_prior_x_true = np.zeros(self.n_mock_datasets)  # ln_prior_x at true model
        self.lnP_theta_true = np.zeros(self.n_mock_datasets)  # lnP_theta at true model

        # Some other things we need if truths were provided that are temporary
        self.ln_prob_x_true_og = np.zeros(self.n_mock_datasets)  # ln_prob_x at true model x

        # Loop over the entire training set and fit every object
        iterator = trange(self.n_mock_datasets, leave=True)

        for imock in iterator:
            iterator.set_description('mcmc> MCMC for mock dataset no: {:d}'.format(imock))

            # Perform the optimization to get the starting point
            observed_data = mock_datasets_to_fit[imock, :]
            x_opt, theta_opt, losses = self.fit_one(observed_data)

            # If the true theta is passed in evaluate lnP_x_true, lnlike_true, ln_prior_x_true
            if self.x_true is not None:
                # First, we need to evaluate the ln_prob_x_true, ln_like_true, ln_prior_x_true
                # ln_prob_x_true is the log probability of the true model x, which means it is the
                # negative of the potential function evaluated at the true model x
                self.ln_prob_x_true[imock] = -self.potential_fn(x_astro=self.x_true[imock, :],
                                                                observed_data=observed_data)

                self.ln_like_true[imock] = self.ln_gaussian_likelihood(x_astro=self.x_true[imock, :],
                                                                       observed_data=observed_data)

                self.ln_prior_x_true[imock] = self.ln_prior(x_astro=self.x_true[imock, :])

                self.ln_prob_x_true_og[imock] = -self.potential_fn(x_astro=self.x_true_og[imock, :],
                                                                   observed_data=observed_data)

                self.lnP_theta_true[imock] = self.lnP_theta(x_astro=self.x_true[imock, :],
                                                            observed_data=observed_data)

            # Split the key
            self.key, subkey = random.split(self.key)

            # do the mcmc for this mock dataset
            imock_mcmc_results = self.mcmc_one(key=subkey,
                                               x_opt=x_opt,
                                               observed_data=observed_data)

            # unpack the results
            x_samples = imock_mcmc_results[0]
            self.x_samples[imock, ...] = np.reshape(x_samples,
                                                    (self.mcmc_num_chains * self.mcmc_nsteps, self.ndim))
            self.samples[imock, ...] = imock_mcmc_results[1]
            self.ln_probs_x[imock, :] = imock_mcmc_results[2]
            # calculate the extra terms for all the samples
            extra_terms_ = np.zeros_like(imock_mcmc_results[2])
            for i in range(imock_mcmc_results[0].shape[0]):
                for j in range(imock_mcmc_results[0].shape[1]):
                    extra_terms_[i, j] = self.extra_term(imock_mcmc_results[0][i, j])
            self.ln_probs_theta[imock, :] = imock_mcmc_results[2] - extra_terms_
            self.neff[imock, :] = imock_mcmc_results[3]
            self.neff_mean[imock] = imock_mcmc_results[4]
            self.sec_per_neff[imock] = imock_mcmc_results[5]
            self.ms_per_step[imock] = imock_mcmc_results[6]
            self.r_hat[imock, :] = imock_mcmc_results[7]
            self.r_hat_mean[imock] = imock_mcmc_results[8]
            self.hmc_num_steps[imock, :] = imock_mcmc_results[9]
            self.hmc_tree_depth[imock, :] = imock_mcmc_results[10]
            self.runtime[imock] = imock_mcmc_results[11]

            # Make some diagnostic plots
            if out_prefix is not None or debug:
                indx = imock
                qa_imock = 'imock_{:03d}'.format(indx)
                walkerfile = out_prefix + '_walkers_' + qa_imock + '.pdf'
                cornerfile = out_prefix + '_corner_' + qa_imock + '.pdf'
                x_cornerfile = out_prefix + '_x-corner_' + qa_imock + '.pdf'
                infer_file = out_prefix + '_infer_' + qa_imock + '.pdf'

                _x_true = self.x_true[imock, :] if self.x_true is not None else None
                _theta_true = self.theta_true[imock, :] if self.theta_true is not None else None
                _ln_prob_x_true = self.ln_prob_x_true[imock] if self.theta_true is not None else None
                # _ln_prob_true = self.ln_prob_true[imock, :] if self.theta_true is not None else None

                # Produce the plots, here is the label we use for this task
                label = self.inferred_plots_dict['params_labels']
                # Walker plot np.swapaxes(x_samples, 0, 1)

                # walker_plot(chain=x_samples,
                #             param_names=label,
                #             truths=self.x_true[imock, :] if self.x_true is not None else None,
                #             probs=self.ln_probs_x[imock, :],
                #             prob_true=_ln_prob_x_true,
                #             walkerfile=walkerfile,
                #             linewidth=0.8)
                # plt.close()
                walker_plot(chain=x_samples,
                            param_names=label,
                            truths=self.x_true[imock, :] if self.x_true is not None else None,
                            walkerfile=walkerfile,
                            linewidth=0.8)
                plt.close()
                # Corner plot in HMC units
                corner_plot(self.x_samples[imock, ...], label,
                            theta_true=_x_true if self.x_true is not None else None,
                            cornerfile=x_cornerfile)
                plt.close()
                # Corner plot in physical units
                corner_plot(self.samples[imock, ...], label,
                            theta_true=self.theta_true[imock, :] if self.theta_true is not None else None,
                            cornerfile=cornerfile)
                plt.close()
                # Inferred model plot
                self.inferred_model_plot(theta_samples=self.samples[imock, ...],
                                         theta_true=self.theta_true[imock, :] if self.theta_true is not None else None,
                                         observed_data=observed_data,
                                         n_posterior_draws=500,
                                         infer_file=infer_file)
                plt.close()

        plt.close('all')

        if out_prefix is not None:
            mcmc_savefile = out_prefix + '.hdf5'
            self.mcmc_save(mcmc_savefile)

    def mcmc_save(self, mcmc_savefile):
        """
        Save the MCMC results to an HDF5 file

        Args:
            mcmc_savefile (str):
                output file for MCMC results

        Returns:

        """

        with h5py.File(mcmc_savefile, 'w') as f:
            group = f.create_group('mcmc')
            # Set the attribute parameters of the MCMC
            group.attrs['mcmc_nsteps'] = self.mcmc_nsteps
            group.attrs['mcmc_num_chains'] = self.mcmc_num_chains
            group.attrs['mcmc_warmup'] = self.mcmc_warmup
            group.attrs['mcmc_dense_mass'] = self.mcmc_dense_mass
            group.attrs['mcmc_max_tree_depth'] = self.mcmc_max_tree_depth
            group.attrs['mcmc_init_perturb'] = self.mcmc_init_perturb
            group.attrs['mcmc_nsteps_tot'] = self.mcmc_nsteps_tot
            # Some other parameters of the run
            group.attrs['n_mock_datasets'] = self.n_mock_datasets
            group.attrs['ndim'] = self.ndim
            # MCMC results
            group.create_dataset('neff', data=self.neff)
            group.create_dataset('neff_mean', data=self.neff_mean)
            group.create_dataset('sec_per_neff', data=self.sec_per_neff)
            group.create_dataset('ms_per_step', data=self.ms_per_step)
            group.create_dataset('r_hat', data=self.r_hat)
            group.create_dataset('r_hat_mean', data=self.r_hat_mean)
            group.create_dataset('hmc_num_steps', data=self.hmc_num_steps)
            group.create_dataset('hmc_tree_depth', data=self.hmc_tree_depth)
            group.create_dataset('runtime', data=self.runtime)
            group.create_dataset('samples', data=self.samples)
            group.create_dataset('x_samples', data=self.x_samples)
            group.create_dataset('ln_probs_x', data=self.ln_probs_x)
            group.create_dataset('ln_probs_theta', data=self.ln_probs_theta)
            # Save the data
            group.create_dataset('autocorrelation_fn_mock_datasets', data=self.autocorrelation_fn_mock_datasets)
            # Some other things related to truths
            if self.theta_true is not None:
                group.create_dataset('ln_prob_x_true', data=self.ln_prob_x_true)
                group.create_dataset('ln_like_true', data=self.ln_like_true)
                group.create_dataset('ln_prior_x_true', data=self.ln_prior_x_true)
                group.create_dataset('theta_true', data=self.theta_true)
                group.create_dataset('x_true', data=self.x_true)
                group.create_dataset('lnP_theta_true', data=self.lnP_theta_true)
                # temporary
                group.create_dataset('ln_prob_x_true_og', data=self.ln_prob_x_true_og)

    # functions to do the MCMC initialization
    def x_minmax(self):
        x_min, x_max = bounded_theta_to_x(self.theta_astro_mins, self.theta_astro_ranges), \
                        bounded_theta_to_x(self.theta_astro_maxs, self.theta_astro_ranges)

        return x_min, x_max

    def mcmc_init_x(self, nwalkers, perturb, x_opt):
        """

        Args:
            nwalkers:
            perturb:
            x_opt:

        Returns:

        """
        # get the ranges in dimensionless space
        x_min, x_max = self.x_minmax()
        delta_x = x_max - x_min

        # create the initial theta values for the walkers
        # this replaces the clipping that was previously done on rectangular prior distributions
        x_init = np.zeros((nwalkers, self.ndim))
        for i in range(nwalkers):
            # propose a new theta_init
            self.key, subkey = random.split(self.key)
            deviates_i = perturb * random.normal(subkey, (1, self.ndim))
            x_init_i = x_opt + deviates_i * delta_x.reshape((1, self.ndim))

            # check that the proposed theta_init is within the bounds of the convex hull
            # repeat until it is, or a max number of 100000 iterations is reached
            n_iteraions = 0
            while not self.point_in_hull(self.x_to_theta(x_init_i.flatten())):
                self.key, subkey = random.split(self.key)
                deviates_i = perturb * random.normal(subkey, (1, self.ndim))
                x_init_i = x_opt + deviates_i * delta_x.reshape((1, self.ndim))
                n_iteraions += 1

                # check if we have reached the max number of iterations
                if n_iteraions > 100000:
                    # print a warning, and break the loop
                    print('Warning: could not find a valid theta_init within the convex hull')
                    break

            # assign the proposed theta_init to the array
            x_init[i, :] = x_init_i

        return x_init.squeeze()

    def inferred_model_plot(self, theta_samples, observed_data, n_posterior_draws, infer_file, theta_true=None):
        """
        Plot the inferred model with the data
        Args:
            theta_samples (ndarray):
                MCMC samples; shape (n_samples, n_dim)
            observed_data (ndarray):
                observed data; shape (n_data, )
            n_posterior_draws (int):
                number of posterior draws to plot
            infer_file (str):
                output file name
            theta_true (ndarray):
                true parameter values; shape (n_dim, )
        """
        # calculate the inferred theta values
        inferred_theta = np.median(theta_samples, axis=0)
        # calculate the inferred model, first emulate all the samples
        # sampled_models = self.mean_emulator.predict(theta_samples)  # ORIGINAL
        sampled_models = self.mean_emulator.predict_unlogged(theta_samples)
        inferred_model = np.median(sampled_models, axis=0)
        # randomly select the sampled models to plot
        self.key, subkey = random.split(self.key)
        sampled_models_to_plot = random.permutation(key=subkey,
                                                    x=sampled_models,
                                                    axis=0,
                                                    independent=False)[:n_posterior_draws]

        # FOR SIMPLICITY, AND SINCE THE EMULATION OF THE MEAN AND THE DIAGONAL OF THE COVARIANCE IS USUALLY REALLY GOOD
        # WE WILL JUST USE THEM TO PLOT THE TRUE MODEL AND THE ERRORS
        mock_error_cov = self.cov_emulator.predict_single(inferred_theta)
        mock_error = np.sqrt(np.diag(mock_error_cov))

        # make the plot
        with plt.style.context(['science', 'no-latex']):
            fig, ax = plt.subplots(1, 1, figsize=(9, 4))

            # randomly sampled models
            for i in range(n_posterior_draws):
                ax.plot(self.inferred_plots_dict['xvalues'], sampled_models_to_plot[i], alpha=0.02, lw=0.5, c='k')
            ax.plot([], c='k', alpha=0.5, lw=1, label='Posterior Draws')

            # the inferred model
            ax.plot(self.inferred_plots_dict['xvalues'], inferred_model, label='Inferred Model', c='blue', lw=2.5)

            if theta_true is not None:

                # maybe I can write code to show the actual true model doing some sort of ngp interpolation,
                # but for now, I will just plot the true model as the emulated one on the true parameters,
                # which is actually really close considering the small percentage error on the emulator
                # true_model_mean_statistic = self.mean_emulator.predict(theta_true).ravel()  # ORIGINAL
                true_model_mean_statistic = self.mean_emulator.predict_unlogged(theta_true).ravel()

                ax.plot(self.inferred_plots_dict['xvalues'], true_model_mean_statistic, '--', lw=2.,
                        label='True Model', c='#FF8225', alpha=0.9)

            # the data with error bars
            ax.errorbar(self.inferred_plots_dict['xvalues'], observed_data, yerr=mock_error,
                        marker='s', markersize=5, label='Mock Data', c='k', ls='None')

            ax.tick_params(axis='both', labelsize=21)

            # ax.legend()
            ax.legend(loc='lower left', fontsize=15)
            ax.set_xlabel(self.inferred_plots_dict['xlabel'], fontsize=25)  # 14
            ax.set_ylabel(self.inferred_plots_dict['ylabel'], fontsize=25)  # 14
            ax.set_ylim(self.inferred_plots_dict['ylim'])
            ax.set_yscale('log')

            plt.savefig(infer_file, bbox_inches='tight')