####################################################################################################
#
# Utility functions mainly for evaluating the emulators' performances
#
####################################################################################################
import numpy as np


def create_correlation_matrix(covariance_matrix):
    """Creates the correlation matrix from the covariance matrix.

    Args:
        covariance_matrix (np.ndarray): The covariance matrix.

    Returns:
        np.ndarray: The correlation matrix.
    """
    # create the correlation matrix
    correlation_matrix = np.zeros_like(covariance_matrix)
    for i in range(covariance_matrix.shape[0]):
        for j in range(covariance_matrix.shape[1]):
            correlation_matrix[i, j] = covariance_matrix[i, j] / np.sqrt(covariance_matrix[i, i] * covariance_matrix[j, j])
    return correlation_matrix


def calculate_cov_diag_error(covariance_matrix, emulated_covariance_matrix):
    """Calculates the diagonal error of an emulated covariance matrix, using the following:

        num[i, j] = np.sqrt(covariance_matrix[i, j] - emulated_covariance_matrix[i, j])

        denom[i, j] = np.sqrt(covariance_matrix[i, i] * covariance_matrix[j, j])

        error[i, j] = num[i, j] / denom[i, j]

        But only on the diagonal elements of the covariance matrix. This gives you the error of the diagonal elements
        relative to the standard deviation represented by the diagonal elements of the covariance matrix true covariance
        matrix.

    Args:
        covariance_matrix (np.ndarray): The covariance matrix.
        emulated_covariance_matrix (np.ndarray): The emulated covariance matrix.

    Returns:
        np.ndarray: The error as described above.
    """
    # create the error matrix
    diag_error = np.diag(calculate_cov_error(covariance_matrix, emulated_covariance_matrix))

    return diag_error


def calculate_cov_error(true_covariance_matrix, emulated_covariance_matrix):
    """Calculates the error of an emulated covariance matrix, using the following:

        error[i, j] = np.abs(true_covariance_matrix[i, j] - emulated_covariance_matrix[i, j]) /
                        np.sqrt(true_covariance_matrix[i, i] * true_covariance_matrix[j, j])

    Args:
        true_covariance_matrix (np.ndarray): The correlation matrix.
        emulated_covariance_matrix (np.ndarray): The emulated correlation matrix.

    Returns:
        np.ndarray: The error as described above.
    """
    # create the error matrix
    error = np.zeros_like(true_covariance_matrix)
    for i in range(true_covariance_matrix.shape[0]):
        for j in range(true_covariance_matrix.shape[1]):
            error[i, j] = np.abs(true_covariance_matrix[i, j] - emulated_covariance_matrix[i, j]) / \
                          np.sqrt(true_covariance_matrix[i, i] * true_covariance_matrix[j, j])
    return error


def calculate_cov_error_optimized(true_covariance_matrix, emulated_covariance_matrix):
    """
    Vectorized version of covariance error calculation.
    """
    diag = np.diag(true_covariance_matrix)
    denom = np.sqrt(np.outer(diag, diag))  # shape (N, N)
    error = np.abs(true_covariance_matrix - emulated_covariance_matrix) / denom
    return error