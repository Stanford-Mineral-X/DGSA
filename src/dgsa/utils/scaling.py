# Copyright (c) 2026 Stanford Mineral-X
# Licensed under the MIT License — see LICENSE file in the root of this repository for details.
"""
Function to scale model responses before computing a distance matrix.
"""

from typing import Literal
from numpy.typing import NDArray
import numpy as np

def scale_responses(
        response_values: NDArray[np.float64],
        method: Literal['raw', 'minmax', 'zscore'] = 'minmax',
        ) -> NDArray[np.float64]:
    """
    Scale each response (column) so that responses with different units
    contribute comparably to the euclidean distance.

    Parameters
    ----------
    response_values : np.ndarray of shape (n_samples, n_responses)
        Model response values

    method : {'raw', 'minmax', 'zscore'}, default = 'minmax'
        - 'raw'    : no scaling
        - 'minmax' : (x - min) / (max - min), each column mapped to [0, 1]
        - 'zscore' : (x - mean) / std, each column has mean 0 and std 1

    Returns
    -------
    response_values_scaled : np.ndarray of shape (n_samples, n_responses)
        Scaled response values. Constant columns are set to 0 for 'minmax' and 'zscore'.
    """

    x = np.asarray(response_values, dtype=float)

    if method == 'raw':
        return x.copy()
    elif method == 'minmax':
        center = x.min(axis=0)
        spread = x.max(axis=0) - center
    elif method == 'zscore':
        center = x.mean(axis=0)
        spread = x.std(axis=0)
    else:
        raise ValueError(f"method must be 'raw', 'minmax' or 'zscore', got '{method}'")

    # avoid division by zero for constant responses
    spread = np.where(spread == 0, 1.0, spread)
    return (x - center) / spread
