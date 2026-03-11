"""Functions for estimating empirical mean and dispersion of genes.

This module provides:
- nb_fit: to fit negative binomial model to count data
- estimate_disp_dist: to estimate dispersion of genes from multiple samples
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom

from .utils import downsample


def nb_fit(x):
    """Fit negative binomial parameters to count data.

    Parameters
    ----------
    x : ndarray
        count data

    Returns
    -------
    pandas Series
        mean (mu) and reciprocal of dispersion (size) of count data
    """
    if np.var(x) < np.mean(x):
        return pd.Series({"size": 1e6, "mu": np.mean(x)})

    try:
        m = np.mean(x)
        v = np.var(x)

        def neg_log_likelihood(params):
            size, mu = params
            return -np.sum(nbinom.logpmf(x, n=size, p=size / (size + mu)))

        initial_guess = [m / (v - m), m]
        bounds = [(0, 1e6), (0, 1e6)]
        result = minimize(
            neg_log_likelihood, initial_guess, bounds=bounds, method="L-BFGS-B"
        )

        if result.success:
            return pd.Series({"size": result.x[0], "mu": m})
        else:
            return pd.Series({"size": np.nan, "mu": np.nan})
    except Exception:
        return pd.Series({"size": np.nan, "mu": np.nan})


def estimate_disp_dist(counts, parallel=True, ncores=None):
    """Estimate mean and dispersion for all samples in data in parallel.

    Parameters
    ----------
    counts : ndarray
        count data matrix
    parallel : bool, optional
        should the samples be processed in parallel, by default True
    ncores : _type_, optional
        with how many cores, by default None

    Returns
    -------
    _pandas DataFrame
        estimates of mean and dispersion for all genes in data
    """
    counts = downsample(counts, parallel=parallel, ncores=ncores)

    if parallel:
        if ncores is None:
            ncores = min(32, (os.cpu_count() or 1) + 4)  # Default to number of CPUs
        with ThreadPoolExecutor(max_workers=ncores) as executor:
            futures = [
                executor.submit(nb_fit, counts[i, :]) for i in range(counts.shape[0])
            ]
            ests = [future.result() for future in as_completed(futures)]
    else:
        ests = [nb_fit(counts[i, :]) for i in range(counts.shape[0])]

    final_ests = pd.DataFrame(ests).dropna()
    return final_ests
