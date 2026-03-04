####################################################################################################
#
# Data loaders for HDF5 datasets produced by the toy pipeline.
#
####################################################################################################

import os
from typing import Dict

import h5py
import numpy as np


_DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "datasets",
    "toy_linear_cov_dataset.h5",
)


class ToyLinearCovLoader:
    """Loader for the toy linear covariance HDF5 dataset.

    Loads the HDF5 file at instantiation and exposes a single method to return
    all arrays as a dictionary. The file is expected to contain datasets
    'theta', 'mu', 'Sigma', 'y_mocks' and a file-level attribute 'x'.

    Attributes:
        path (str): Path to the HDF5 file that was loaded.
    """

    def __init__(self, path: str = _DEFAULT_DATASET_PATH):
        """Load the toy linear covariance dataset from an HDF5 file.

        Args:
            path: Path to the HDF5 file. Defaults to the bundled
                toy_linear_cov_dataset.h5 in this package.
        """
        self.path = path
        with h5py.File(path, "r") as f:
            self._theta = np.asarray(f["theta"])
            self._mu = np.asarray(f["mu"])
            self._Sigma = np.asarray(f["Sigma"])
            self._y_mocks = np.asarray(f["y_mocks"])
            self._x = np.asarray(f.attrs["x"])

    def get_data(self) -> Dict[str, np.ndarray]:
        """Return all dataset arrays as a dictionary.

        Returns:
            Dictionary with keys 'theta', 'mu', 'Sigma', 'y_mocks', and 'x',
            and values as the corresponding numpy arrays.
        """
        return {
            "theta": self._theta,
            "mu": self._mu,
            "Sigma": self._Sigma,
            "y_mocks": self._y_mocks,
            "x": self._x,
        }
