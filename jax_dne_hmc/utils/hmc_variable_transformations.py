import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
## TESTING Disable JIT for now
#from jax.config import config
#config.update('jax_disable_jit', True)
#TESTING
from IPython import embed

@jit 
def bounded_variable_lnPr(x):
    """
    This is the prior probability, Pr(theta) for a parameter theta distributed uniformly on the range (theta_0, theta_1). Since
    we are working with the transformed parameter x, we need to compute the prior probability for x. This is just a constant
    which we can drop, so we write a dummy function here that just returns zero. 
    
    Args:
        x (jax.numpy array):
            HMC parameter vector. Shape (nsamples, ndim) or (ndim,)
    Returns:
        lnPr (jax.numpy array):
            Log prior probability at the parameter vector. Shape (nsamples, ndim) or (ndim,)
    """
    
    return 0*x


@jit
def bounded_theta_to_x(theta, theta_ranges):
    """
    Transform a bounded paramter vector theta into an unbounded parameter vector x using a logit transformation.
    This is the single vector function called by the vectorized function in base.

    Args:
        theta (jax.numpy.ndarray):
            Parameter vector with shape=(n_params,)
        theta_range (list):
            List of length n_params containing 2-d tuples, where each tuple is the range of the parameter.
            The first element of the tuple is the lower bound, and the second element is the upper bound.

    Returns:
        x (jax.numpy.ndarray):
            Transformed parameter vector with shape=(n_params,)

    """

    _theta = jnp.atleast_1d(theta)
    n_params = _theta.shape[0]
    x = jnp.zeros(n_params)

    for i in range(n_params):
        x = x.at[i].set(_bounded_theta_to_x(_theta[i], theta_ranges[i]))

    return jnp.array(x)


@jit
def _bounded_theta_to_x(theta_element, theta_range):
    """
    Transform an element of a bounded parameter vector theta into an element of an unbounded parameter vector x using a
    logit transformation. This is the single element function called by the functions above.

    Args:
        theta_element (float):
            Element of a parameter vector.
        theta_range (tuple):
            A tuple of length=2 specifying the range of the parameter.
            The first element of the tuple is the lower bound, and the second element is the upper bound.

    Returns:
        x_element (float):
            Transformed parameter vector element.

    """

    return jax.scipy.special.logit(
        jnp.clip((theta_element - theta_range[0])/(theta_range[1] - theta_range[0]), a_min=1e-7, a_max=1.0 - 1e-7))


@jit
def x_to_bounded_theta(x, theta_ranges):
    """
    Transform an unbounded parameter vector x into a bounded paramter vector theta using a sigmoid transformation.
    This is the single vector function called by the vectorized function in base.

    Args:
        x (jax.numpy.ndarray):
            Transformed parameter vector with shape=(n_params,)
        theta_ranges:
            List of length n_params containing 2-d tuples, where each tuple is the range of the parameter.
            The first element of the tuple is the lower bound, and the second element is the upper bound.

    Returns:
        theta (jax.numpy.ndarray):
             Parameter vector with shape=(n_params,)
    """

    _x = jnp.atleast_1d(x)
    n_params = _x.shape[0]
    theta = jnp.zeros(n_params)

    for i in range(n_params):
        theta = theta.at[i].set(_x_to_bounded_theta(_x[i], theta_ranges[i]))

    return jnp.array(theta)


@jit
def _x_to_bounded_theta(x_element, theta_range):
    """
    Transform an element of an unbounded parameter vector x into an elements of a paramter vector theta using a
    sigmoid transformation. This is the single element function called by the functions above.

    Args:
        x_element (float):
            Element of a transformed parameter vector
        theta_range (tuple):
            A tuple of lenegth=2 specifying the range of the parameter.
            The first element of the tuple is the lower bound, and the second element is the upper bound.

    Returns:
        theta_element (float):
             Element of the parameter vector theta.
    """

    return theta_range[0] + (theta_range[1] - theta_range[0])*jax.nn.sigmoid(x_element)