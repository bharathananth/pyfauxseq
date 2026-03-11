"""Utility functions to downsample and normalize count data.

This module provides:
- drop_counts: to downsample counts to a certain depth
- downsample: to downsample counts in multiple samples to a common depth
- load_dataset: to load csv data
- normalize_counts: to normalize count data using median of ratios
- median_of_ratios: to estimate median of ratios for the samples
"""

import importlib.resources
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from pyfauxseq import data


def drop_counts(y, N):
    """Downsample count data for a single sample such that total counts equals N.

    Parameters
    ----------
    y : array-like
        count data for one sample
    N : int
        target total counts

    Returns
    -------
    array
        downsampled counts

    Raises
    ------
    ValueError
        if target counts exceeds actual sum of counts
    """
    total = np.sum(y)

    if total < N:
        raise ValueError("N is larger than the total counts.")

    if total == N:
        return y
    else:
        all_reads = np.repeat(np.arange(len(y)), y)
        removed = np.random.choice(all_reads, N, replace=False)
        return np.bincount(removed, minlength=len(y))


def downsample(counts, parallel=True, ncores=None):
    """Downsample multiple samples in parallel.

    Parameters
    ----------
    counts : ndarray
        count data to be downsampled (n_genes * n_samples)
    parallel : bool, optional
        should samples be processed in parallel, by default True
    ncores : int | None, optional
        number of cores to be used, by default None (use all available)

    Returns
    -------
    ndarray
        downsampled count data matrix
    """
    min_lib_size = np.min(np.sum(counts, axis=0))

    if parallel:
        if ncores is None:
            ncores = min(32, (os.cpu_count() or 1) + 4)  # Default to number of CPUs
        with ThreadPoolExecutor(max_workers=ncores) as executor:
            futures = [
                executor.submit(drop_counts, counts[:, i], min_lib_size)
                for i in range(counts.shape[1])
            ]
            downsampled_counts = np.column_stack(
                [future.result() for future in as_completed(futures)]
            )
    else:
        downsampled_counts = np.column_stack(
            [drop_counts(counts[:, i], min_lib_size) for i in range(counts.shape[1])]
        )

    return downsampled_counts


def load_dataset(filename):
    """Read data from files (currently only csv supported).

    Parameters
    ----------
    filename : str
        filename to be read

    Returns
    -------
    pandas DataFrame
        file contents as a DataFrame

    Raises
    ------
    ValueError
        if unsupported file format (not csv)
    """
    with importlib.resources.path(data, filename) as data_path:
        if ".csv" in filename:
            return pd.read_csv(data_path)
        else:
            raise ValueError("Unsupported file format")


def normalize_counts(counts, log=True):
    """Normalize counts using median of ratios with optional log transform.

    Parameters
    ----------
    counts : pandas DataFrame
        count data to be normalized
    log : bool, optional
        should the normalized counts be log2 transformed, by default True

    Returns
    -------
    pandas DataFrame
        normalized count data
    """
    norm_counts = counts / np.sum(counts, axis=0) / median_of_ratios(counts) * 1e6
    if log:
        norm_counts = np.log2(1 + norm_counts)
    return norm_counts


def median_of_ratios(counts):
    """Compute median of ratios for each sample.

    Parameters
    ----------
    counts : pandas DataFrame
        count data matrix (n_genes * n_samples)

    Returns
    -------
    ndarray
        median of ratios
    """
    counts = counts.to_numpy()
    return 2 ** np.ma.median(
        np.ma.log2(counts) - np.ma.mean(np.ma.log2(counts), axis=1, keepdims=True),
        axis=0,
    )
