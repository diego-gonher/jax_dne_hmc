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
import corner
import arviz as az
import h5py
import optax
from tqdm.auto import trange
# from scipy.optimize import minimize

from functools import partial
import time
from IPython import embed

from jax_dne_hmc.utils.hmc_variable_transformations import bounded_theta_to_x, x_to_bounded_theta  # variable transformations
from jax_dne_hmc.utils.hmc_variable_transformations import bounded_variable_lnPr  # prior

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

    def __init__(self, theta_prior_ranges, mean_emulator, covar_emulator, dataset_loader=None,
                 opt_nsteps=150, opt_lr=0.01, mcmc_nsteps=1000, mcmc_num_chains=4, mcmc_warmup=1000,
                 mcmc_init_perturb=0.05,  mcmc_max_tree_depth=10, mcmc_dense_mass=True, key=random.PRNGKey(42)):
        """
        Initialize the HMCInference class.

        Args:
            theta_prior_ranges (list):
                List of length n_params containing 2-d tuples, where each tuple is the range of the parameter.
                The first element of the tuple is the lower bound, and the second element is the upper bound.
            mean_emulator (TrainerModule):
                An instance of the MeanEmulator class.
            covar_emulator (TrainerModule):
                The emulator used for the covariance of the autocorrelation function. 
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
        self.ndim = len(theta_prior_ranges)
        self.theta_prior_ranges = theta_prior_ranges
        self.theta_inits = tuple([np.mean([tup[0], tup[1]]) for tup in theta_prior_ranges])  # mean of the ranges
        self.theta_prior_mins = jnp.array([par_range[0] for par_range in self.theta_prior_ranges])
        self.theta_prior_maxs = jnp.array([par_range[1] for par_range in self.theta_prior_ranges])

        # Set parameter priors in the HMC latent space (Smoothed Box Prior is a differentiable uniform distribution)
        self.x_priors = [bounded_variable_lnPr]*self.ndim

        # Store the emulators and the dataset loader
        self.mean_emulator = mean_emulator
        self.covar_emulator = covar_emulator
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
        self.observed_datasets = None
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
    def ln_prior(self, x_param):
        """
        Compute the prior on the theta parameters transformed to the latent 
        HMC space.

        Args:
            x_param (ndarray): shape = (nastro,)
                dimensionless (in the latent HMC space) parameter vector

        Returns:
            prior (float):
                Prior on these model parameters
        """

        prior = 0.0
        for x_, x_pri in zip(x_param, self.x_priors):
            prior += x_pri(x_)

        return prior

    # function that uses the emulators to get the mean and its corresponding covariance matrix
    @partial(jit, static_argnums=(0,))
    def get_mean_and_covar(self, theta):
        """
        Returns the mean of the obesrvable and its corresponding covariance matrix given the model
        parameters.

        Args:
            theta (array): theta parameters in physical units. Shape (ndim,)

        Returns:
            (mean, covariance_matrix)
        """
        # get the mean from the neural emulator
        mean = self.laf_mean_emulator.predict(theta).ravel()  # Is ravel is needed?
        # get the covariance matrix from the neural emulator
        covariance_matrix = self.laf_cov_emulator.predict_single(theta)

        return mean, covariance_matrix

    # Gaussian likelihood
    @partial(jit, static_argnums=(0,))
    def ln_gaussian_likelihood(self, x_param, observation):
        """
        Natural logarithm of the gaussian likelihood.

        Args:
            x_param (jnp.array): value of parameters in dimensionless latent space. Shape of (ndims,).
            observation (jnp.array): observed data that the inference is being done for.

        Returns:
            float: ln of gaussian likelihood.
        """
        # convert to physical units
        theta = self.x_to_theta(x_param)
        # obtain the mean autocorrelation model and its corresponding covariance matrix
        mean, covariance_matrix = self.get_mean_and_covar(theta=theta)
        # compute and return the gaussian likelihood
        return jax.scipy.stats.multivariate_normal.logpdf(x=observation,
                                                          mean=mean,
                                                          cov=covariance_matrix)

    # Now the potential function and its helper used for the numpyro NUTS sampler
    @partial(jit, static_argnums=(0,))
    def potential_fn(self, x_param, observation):
        """
            Potential function for the MCMC.

            Args:
                x_param (jnp.array): value of parameters in dimensionless latent space. Shape of (ndims,).
                observation (jnp.array): observed data that the inference is being done for.

            Returns:
                float: potential.
            """
        # ln(prior) calculated
        lnPrior = self.ln_prior(x_param)
        # ln(likelihood) calculated
        lnL = self.ln_gaussian_likelihood(x_param, observation)
        # Calculate the potential = -(ln(likelihood) + ln(prior)) \propto -ln(posterior)
        potential = -(lnPrior + lnL)
        return potential

    def potential_fn_numpyro(self, observation):
        """
        Wrapper for potential function to be used with numpyro. This allows one to call numpyro HMC with
        given observation since the NumPyro HMC sampler potential function API only allows one to pass
        a function that takes a single argument.

        Args:
            observation (array): observed data that the inference is being done for.

        Returns:
            potentical_function (function):
                The potential function to be used with NumPyro's HMC sampler.

        """
        return partial(self.potential_fn, observation=observation)

    # Quick function that does a quick fit to use as init values for the HMC
    def fit_one(self, observation):
        """
        Fit a single observation using the Adam optimizer

        Args:
            observation (ndarray): shape (ndimobs,)
                data that we are doing the inference for.

        Returns:
            x_out (ndarray): shape (ndim,)
                best fit dimensionless parameter vector
            theta_out (ndarray): shape (ndim,)
                best fit parameter vector in physical units
            losses (ndarray): shape (opt_nsteps,)
                loss function values at each iteration step
        """

        # Initialize the parameters for the new fit, just use the prior it is safer
        x = self.theta_to_x(self.theta_inits)
        optimizer = optax.adam(self.opt_lr)
        opt_state = optimizer.init(x)
        losses = np.zeros(self.opt_nsteps)
        # Optimization loop for fitting input flux
        iterator = trange(self.opt_nsteps, leave=False)
        best_loss = np.inf  # Models are only saved if they reduce the validation loss
        for i in iterator:
            losses[i], grads = jax.value_and_grad(self.potential_fn, argnums=0)(x, observation)
            if losses[i] < best_loss:
                x_out = x.copy()
                theta_out = self.x_to_theta(x_out)
                best_loss = losses[i]
            updates, opt_state = optimizer.update(grads, opt_state)
            x = optax.apply_updates(x, updates)

        return x_out, theta_out, losses

    # Now the HMC function
    def mcmc_one(self, key, x_opt, observation):
        """
        HMC routine for a single quasar

        Args:
            key (JAX PRNG key)
                pseudo-random number generator key
            x_opt (ndarray): shape (ndim,)
                best fit dimensionless parameter vector
            observation (ndarray): shape (ndimobs,)
                observation that the inference is being done for.

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
        nuts_kernel = NUTS(potential_fn=self.potential_fn_numpyro(observation),
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

    def mcmc(self, observation_datasets_to_fit, theta_true,
             out_prefix=f'data/multiple_inferences/', debug=False):
        """Run the MCMC sampler.

        Args:
            mock_datasets_to_fit (ndarray): autocorrelation function data; shape (n_mock_datasets, velocity bins).
            theta_true (ndarray): true values of the parameters used to generate the mock datasets; shape
                (n_mock_datasets, ndims). The first column is the mfp and the second is the mean flux.
            out_prefix (str): Prefix for the output files. Default is 'data/multiple_inferences/'.
            debug (bool): Show some plots that can be used as diagnostics.

        """
        # Now we set some class attributes used to save the results
        self.n_mock_datasets = observation_datasets_to_fit.shape[0]
        self.theta_true = jnp.atleast_2d(theta_true) if theta_true is not None else None
        self.observed_datasets = observation_datasets_to_fit
        self.x_true = jnp.atleast_2d(self.theta_to_x(theta_true)) if theta_true is not None else None

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
            observation = observation_datasets_to_fit[imock, :]
            x_opt, theta_opt, losses = self.fit_one(observation)

            # If the true theta is passed in evaluate lnP_x_true, lnlike_true, ln_prior_x_true
            if self.x_true is not None:
                # First, we need to evaluate the ln_prob_x_true, ln_like_true, ln_prior_x_true
                # ln_prob_x_true is the log probability of the true model x, which means it is the
                # negative of the potential function evaluated at the true model x
                self.ln_prob_x_true[imock] = -self.potential_fn(x_param=self.x_true[imock, :],
                                                                observation=observation)

                self.ln_like_true[imock] = self.ln_gaussian_likelihood(x_param=self.x_true[imock, :],
                                                                       observation=observation)

                self.ln_prior_x_true[imock] = self.ln_prior(x_param=self.x_true[imock, :])

                self.ln_prob_x_true_og[imock] = -self.potential_fn(x_param=self.x_true_og[imock, :],
                                                                   observation=observation)

            # Split the key
            self.key, subkey = random.split(self.key)

            # do the mcmc for this mock dataset
            imock_mcmc_results = self.mcmc_one(key=subkey,
                                               x_opt=x_opt,
                                               observation=observation)

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
                cornerfile = out_prefix + '_corner_' + qa_imock + '.pdf'
                x_cornerfile = out_prefix + '_x_corner_' + qa_imock + '.pdf'

                _x_true = self.x_true[imock, :] if self.x_true is not None else None
                _theta_true = self.theta_true[imock, :] if self.theta_true is not None else None
                # embed(header='mcmc> debug here')
                _ln_prob_x_true = self.ln_prob_x_true[imock] if self.theta_true is not None else None
                # _ln_prob_true = self.ln_prob_true[imock, :] if self.theta_true is not None else None

                # Produce the plots, make it as generic as possible
                plt.close()
                # Corner plot in HMC units
                corner_figure_x = corner.corner(self.x_samples[imock, ...],
                                                truths=_x_true if self.x_true is not None else None)
                corner_figure_x.savefig(x_cornerfile, bbox_inches="tight")
                plt.close(corner_figure_x)
                # Corner plot in physical units
                corner_figure_theta = corner.corner(self.samples[imock, ...],
                                                    truths=self.theta_true[imock, :] if self.theta_true is not None else None)
                corner_figure_theta.savefig(cornerfile, bbox_inches="tight")
                plt.close(corner_figure_theta)

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
            group.create_dataset('observed_datasets', data=self.observed_datasets)
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
