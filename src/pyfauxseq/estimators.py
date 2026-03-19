"""Functions for estimating empirical mean and dispersion of genes.

This module provides:
- nb_fit: to fit negative binomial model to count data
- estimate_disp_dist: to estimate dispersion of genes from multiple samples
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures._base import Future

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.core.series import Series
from scipy.optimize import minimize
from scipy.stats import nbinom

from .utils import downsample


def nb_fit(x: NDArray) -> pd.Series:
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
        m: float = np.mean(x)
        v: float = np.var(x)

        def neg_log_likelihood(params):
            size, mu = params
            return -np.sum(nbinom.logpmf(x, n=size, p=size / (size + mu)))

        initial_guess: list[float] = [m / (v - m), m]
        bounds: list[tuple[int, int | float]] = [(0, 1e6), (0, 1e6)]
        result = minimize(
            neg_log_likelihood, initial_guess, bounds=bounds, method="L-BFGS-B"
        )

        if result.success:
            return pd.Series({"size": result.x[0], "mu": m})
        else:
            return pd.Series({"size": np.nan, "mu": np.nan})
    except Exception:
        return pd.Series({"size": np.nan, "mu": np.nan})


def estimate_disp_dist(
    counts: NDArray, parallel: bool = True, ncores: int | None = None
) -> pd.DataFrame:
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
    pandas DataFrame
        estimates of mean and dispersion for all genes in data
    """
    counts: NDArray = downsample(counts, parallel=parallel, ncores=ncores)

    if parallel:
        if ncores is None:
            ncores: int = min(32, (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=ncores) as executor:
            futures: list[Future[Series]] = [
                executor.submit(nb_fit, counts[i, :]) for i in range(counts.shape[0])
            ]
            ests: list[Series] = [future.result() for future in as_completed(futures)]
    else:
        ests: list[Series] = [nb_fit(counts[i, :]) for i in range(counts.shape[0])]

    final_ests: pd.DataFrame = pd.DataFrame(ests).dropna()
    return final_ests
