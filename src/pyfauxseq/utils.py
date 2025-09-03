from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.resources
import numpy as np
import os
import pandas as pd
from pyfauxseq import data

def drop_counts(y, N):
    total = np.sum(y)

    if total < N:
        raise ValueError("N is larger than the total number of objects.")

    if total == N:
        return y
    else:
        all_reads = np.repeat(np.arange(len(y)), y)
        removed = np.random.choice(all_reads, N, replace=False)
        return np.bincount(removed, minlength=len(y))

def downsample(counts, parallel=True, ncores=None):
    min_lib_size = np.min(np.sum(counts, axis=0))

    if parallel:
        if ncores is None:
            ncores = min(32, (os.cpu_count() or 1) + 4)  # Default to number of CPUs
        with ThreadPoolExecutor(max_workers=ncores) as executor:
            futures = [executor.submit(drop_counts, counts[:, i], min_lib_size) for i in range(counts.shape[1])]
            downsampled_counts = np.column_stack([future.result() for future in as_completed(futures)])
    else:
        downsampled_counts = np.column_stack([drop_counts(counts[:, i], min_lib_size) for i in range(counts.shape[1])])

    return downsampled_counts

def load_dataset(filename):
    with importlib.resources.path(data, filename) as data_path:
        if ".csv" in filename:
            return pd.read_csv(data_path)
        else:
            raise ValueError("Unsupported file format")

def normalize_counts(counts, log=True):
    norm_counts = counts/np.sum(counts, axis=0)/median_of_ratios(counts)*1e6
    if log:
        norm_counts = np.log2(1 + norm_counts)
    return(norm_counts)

def median_of_ratios(counts):
    counts = counts[counts.sum(axis=1)>0]
    return(2 ** np.nanmedian(np.log2(counts) - \
        np.nanmean(np.log2(counts), axis=1, keepdims=True), axis=0))
    
