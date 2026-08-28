"""Functions for generating rhythmic transcriptomic data.

This module currently provides:
- generate_rhythmic_rnaseq: to generate data under one condition
- generate_diffrhythmic_rnaseq: to generate data under two conditions
"""

import numpy as np
import pandas as pd
from numpy.random._generator import Generator
from numpy.typing import ArrayLike, NDArray
from scipy.stats import nbinom

from .utils import load_dataset


def generate_rhythmic_rnaseq(
    t: ArrayLike = (0, 4, 8, 12, 16, 20),
    reps: int = 1,
    period: int = 24,
    n_genes: int = 10000,
    rhy_frac: float = 0.1,
    min_A_effect: float = 0.2,
    A_spread: float = 0.5,
    emp_dist: dict | str = "liver",
    depth: int = 40000000,
    lib_size_var: tuple[float, float] = (0.8, 1.2),
    seed=None,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic rhythmic RNA-seq count data in one condition.

    This function generates artificial timeseries transcriptomic data under one
    condition, with rhythmically expressed genes and empirically estimated
    variability between replicates/samples.

    Parameters
    ----------
    t : numpy array-like, optional
        time points at which samples are generated, by default
        (0, 4, 8, 12, 16, 20)
    reps : int or array-like, optional
        number of replicates at each time point, by default 1
    period : int, optional
        period of the rhythmic genes, by default 24
    n_genes : int, optional
        number of genes in the dataset, by default 10000
    rhy_frac : float, optional
        fraction of genes that are rhythmic, by default 0.1
    min_A_effect : float, optional
        minimum amplitude (in log2 fold) of rhythmic genes, by default 0.2
    A_spread : int, optional
        mean of the exponential distribution of rhythmic gene amplitudes (in
        log2 fold), by default 0.5
    emp_dist : dictionary with keys 'mu' and 'size'| str, optional
        empirical mean (mu) and size (1/dispersion) values for a corpus of
        genes or name of a file containing the dictionary, by default "liver"
        (read from mouse liver dataset)
    depth : int, optional
        average sequencing depth of different samples, by default 4e7
    lib_size_var : tuple of floats, optional
        window of variability of sequencing depth of samples about 'depth',
        by default (0.8, 1.2)
    seed : int, optional
        seed to ensure reproducible datasets, by default None

    Returns
    -------
    dictionary
        counts: pandas DataFrame (n_genes * n_samples) with count data
        params: pandas DataFrame (n_rhythmic_genes * 3) with identity of
        rhythmic genes, and their amplitudes (A) and phases (phi).
        exp_design: pandas DataFrame (n_samples * 1) with time labels of
        individual samples

    Raises
    ------
    ValueError
        if 'reps' is not an int or has different length than 't'
    FileNotFoundError
        if file containing 'emp_dist' does not exist.
    ValueError
        if provided 'emp_dist' is invalid

    Notes
    -----
    We extended generative of model of gene expression of Soneson & Delorenz
    [SD]_ to also include rhythmic genes.

    References
    ----------
    .. [SD] Soneson C, Delorenzi M. A comparison of methods for differential
    expression analysis of RNA-seq data. BMC Bioinformatics. 2013;14: 91.
    """
    rng: Generator = np.random.default_rng(seed)

    if not isinstance(reps, int) and (
        not isinstance(reps, list) or len(reps) != len(t)
    ):
        raise ValueError("Length of reps must be 1 or the same as length of t")

    if isinstance(emp_dist, str):
        match emp_dist:
            case "liver":
                emp_dist: pd.DataFrame = load_dataset("Mm_liver_LD_NC.csv.gz")
            case "multitissue":
                emp_dist: pd.DataFrame = load_dataset("Mm_multitissue_LD_ALF.csv.gz")
            case _:
                emp_dist: pd.DataFrame = load_dataset("Mm_liver_LD_NC.csv.gz")
    elif not isinstance(emp_dist, pd.DataFrame):
        raise ValueError("The provided emp_dist is invalid.")

    if emp_dist.shape[1] > 2:
        group_cols = list(pd.Index.difference(emp_dist.columns, ["size", "mu"]))

        unique_groups = emp_dist[group_cols].drop_duplicates()
        sampled_row = unique_groups.iloc[rng.choice(len(unique_groups))]

        selected_key = (
            sampled_row.iloc[0] if len(group_cols) == 1 else tuple(sampled_row)
        )

        by_param = group_cols[0] if len(group_cols) == 1 else group_cols
        emp_dist = emp_dist.groupby(by=by_param).get_group(selected_key)

    t: NDArray = np.array(t)
    t: NDArray = np.repeat(t, reps)
    N: int = len(t)

    G_rhy: NDArray = rng.uniform(size=n_genes) <= rhy_frac

    A: NDArray = min_A_effect + rng.exponential(A_spread, n_genes)

    A[~G_rhy] = 0.0
    phi: NDArray = rng.uniform(size=n_genes) * 2 * np.pi * G_rhy
    params: pd.DataFrame = pd.DataFrame(
        {
            "id": [f"g{i + 1}" for i in np.where(G_rhy)[0]],
            "A": A[G_rhy],
            "phi": phi[G_rhy],
        }
    )

    t_pattern: NDArray = np.cos(phi)[:, None] @ np.cos(
        2 * np.pi * t[None, :] / period
    ) + np.sin(phi)[:, None] @ np.sin(2 * np.pi * t[None, :] / period)

    lib_size_fct: NDArray = rng.uniform(lib_size_var[0], lib_size_var[1], N)
    lib_size: NDArray = lib_size_fct * depth

    draw: NDArray = rng.choice(emp_dist.shape[0], n_genes, replace=True)
    lambda_: NDArray = emp_dist["mu"].values[draw].reshape(-1, 1)

    size: NDArray = emp_dist["size"].values[draw].reshape(-1, 1)

    lambda_: NDArray = np.repeat(lambda_, N, axis=1) * 2 ** (A[:, None] * t_pattern)

    mu: NDArray = lambda_ / (np.sum(lambda_, axis=0) / lib_size)

    counts: NDArray[np.int_] = nbinom.rvs(
        n=size, p=size / (mu + size), random_state=seed
    )

    exp_design = pd.DataFrame(
        {
            "time": t,
        },
        index=pd.Index(
            "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), 5)) for _ in range(N)
        ),
    )

    return {
        "counts": pd.DataFrame(
            counts,
            index=pd.Index([f"g{i + 1}" for i in range(n_genes)]),
            columns=exp_design.index,
        ),
        "params": params,
        "exp_design": exp_design,
    }


def generate_diffrhythmic_rnaseq(
    t: ArrayLike = (0, 4, 8, 12, 16, 20),
    reps: int = 1,
    period: int = 24,
    n_genes: int = 10000,
    rhy_frac: float = 0.1,
    DR_probs: ArrayLike = (1.25, 1.25, 1.25, 1.25),
    min_A_effect: float = 0.2,
    A_spread: float = 0.5,
    DE_frac: float = 0.1,
    min_DE_effect: float = 0.5,
    DE_spread: float = 0.5,
    groups: tuple[str, str] = ("ctrl", "expt"),
    emp_dist: pd.DataFrame | str = "liver",
    depth: int = 40000000,
    lib_size_var: tuple[float, float] = (0.8, 1.2),
    seed=None,
):
    """Generate synthetic rhythmic RNA-seq count data in two conditions.

    This function generates artificial timeseries transcriptomic data under two
    conditions, which includes both differentially rhythmic and differential
    expressed genes, as well as empirically estimated variability between
    replicates/samples.

    Parameters
    ----------
    t : numpy array, optional
        time points at which samples are generated, by default
        np.arange(0, 21, 4)
    reps : int or array-like, optional
        number of replicates at each time point, by default 1
    period : int, optional
        period of the rhythmic genes, by default 24
    n_genes : int, optional
        number of genes in the dataset, by default 10000
    rhy_frac : float, optional
        fraction of genes that are rhythmic, by default 0.1
    DR_probs : numpy array, optional
        determines the relative average number of ("same", "gain", "loss",
        "change") group elements. The larger the numbers, more concentrated are
        the actual numbers around the averages, by default (1.25, 1.25, 1.25,
        1.25)
    min_A_effect : float, optional
        minimum amplitude (in log2 fold) of rhythmic genes, by default 0.2
    A_spread : float, optional
        mean of the exponential distribution of rhythmic gene amplitudes (in
        log2 fold), by default 0.5
    DE_frac : float, optional
        fraction of differentially expressed (DE) genes, by default 0.1
    min_DE_effect : float, optional
        minimum log2 fold change in expression of DE genes, by default 0.5
    DE_spread : float, optional
        mean of the exponential distribution of DE fold changes, by default 0.5
    groups : tuple, optional
        labels for the two groups/conditions, by default ("ctrl", "expt")
    emp_dist : dictionary with keys 'mu' and 'size'| str, optional
        empirical mean (mu) and size (1/dispersion) values for a corpus of
        genes or name of a file containing the dictionary, by default "liver"
        (read from mouse liver dataset)
    depth : int, optional
        average sequencing depth of different samples, by default 4e7
    lib_size_var : tuple of floats, optional
        window of variability of sequencing depth of samples about 'depth',
        by default (0.8, 1.2)
    seed : int, optional
        seed to ensure reproducible datasets, by default None

    Returns
    -------
    dictionary
        counts: pandas DataFrame (n_genes * n_samples * n_groups) with count
            data
        params: pandas DataFrame (n_genes * 7) with identity of gene,
            differential rhythmicity category, amplitude (A) in two groups, phases
            (phi) in the two groups, and DE effect size.
        exp_design: pandas DataFrame ((2*n_samples) * 2) with time labels for
            individual samples in each group

    Raises
    ------
    ValueError
        if 'reps' is not an int or has different length than 't'
    FileNotFoundError
        if file containing 'emp_dist' does not exist.
    ValueError
        if provided 'emp_dist' is invalid

    Notes
    -----
    We extended the generative of model of differential gene expression of
    Soneson & Delorenz [SD]_ to also include differentially rhythmic genes.

    References
    ----------
    .. [SD] Soneson C, Delorenzi M. A comparison of methods for differential
    expression analysis of RNA-seq data. BMC Bioinformatics. 2013;14: 91.
    """
    rng: Generator = np.random.default_rng(seed)

    if not isinstance(reps, int) and (
        not isinstance(reps, list) or len(reps) != len(t)
    ):
        raise ValueError("Length of reps must be 1 or the same as length of t")

    if isinstance(emp_dist, str):
        match emp_dist:
            case "liver":
                emp_dist: pd.DataFrame = load_dataset("Mm_liver_LD_NC.csv.gz")
            case "multitissue":
                emp_dist: pd.DataFrame = load_dataset("Mm_multitissue_LD_ALF.csv.gz")
            case _:
                emp_dist: pd.DataFrame = load_dataset("Mm_liver_LD_NC.csv.gz")
    elif not isinstance(emp_dist, pd.DataFrame):
        raise ValueError("The provided emp_dist is invalid.")

    if emp_dist.shape[1] > 2:
        group_cols = list(pd.Index.difference(emp_dist.columns, ["size", "mu"]))

        unique_groups = emp_dist[group_cols].drop_duplicates()
        sampled_row = unique_groups.iloc[rng.choice(len(unique_groups))]

        selected_key = (
            sampled_row.iloc[0] if len(group_cols) == 1 else tuple(sampled_row)
        )

        by_param = group_cols[0] if len(group_cols) == 1 else group_cols
        emp_dist = emp_dist.groupby(by=by_param).get_group(selected_key)

    t: NDArray = np.repeat(t, reps)
    N: int = len(t)

    G_rhy: NDArray = rng.uniform(size=n_genes) <= rhy_frac
    prior_p = rng.gamma(shape=DR_probs, scale=1.0, size=4)
    prior_p = prior_p / prior_p.sum()
    DR_counts = rng.multinomial(G_rhy.sum(), prior_p, 1).squeeze()
    DR_classes = np.array(["gain", "loss", "change", "same"])
    DR_groups: NDArray = DR_classes[np.repeat(np.arange(4), DR_counts)]

    A: NDArray = min_A_effect + rng.exponential(A_spread, (n_genes, 2))
    A[~G_rhy, :] = 0.0
    phi: NDArray = rng.uniform(size=(n_genes, 2)) * 2 * np.pi * G_rhy.reshape(-1, 1)

    G_rhy_index: NDArray = np.where(G_rhy)[0]
    for i in range(len(DR_groups)):
        if DR_groups[i] == "gain":
            A[G_rhy_index[i], 0] = 0.0
        elif DR_groups[i] == "loss":
            A[G_rhy_index[i], 1] = 0.0
        elif DR_groups[i] == "same":
            A[G_rhy_index[i], 0] = A[G_rhy_index[i], 1]
            phi[G_rhy_index[i], 0] = phi[G_rhy_index[i], 1]

    params: pd.DataFrame = pd.DataFrame(
        {
            "id": [f"g{i + 1}" for i in np.where(G_rhy)[0]],
            "category": DR_groups,
            "A_1": A[G_rhy, 0],
            "A_2": A[G_rhy, 1],
            "phi_1": phi[G_rhy, 0],
            "phi_2": phi[G_rhy, 1],
        }
    )
    params.columns: list[str] = [
        "id",
        "category",
        "A_ctrl",
        "A_expt",
        "phi_ctrl",
        "phi_expt",
    ]

    t_pattern: NDArray = np.hstack(
        [
            np.cos(phi[:, [0]]) @ np.cos(2 * np.pi * t[None, :] / period)
            + np.sin(phi[:, [0]]) @ np.sin(2 * np.pi * t[None, :] / period),
            np.cos(phi[:, [1]]) @ np.cos(2 * np.pi * t[None, :] / period)
            + np.sin(phi[:, [1]]) @ np.sin(2 * np.pi * t[None, :] / period),
        ]
    )

    G_DE: NDArray = rng.uniform(size=n_genes) <= DE_frac
    DE_effects: NDArray = (
        min_DE_effect + rng.exponential(DE_spread, n_genes)
    ) * np.sign(2 * rng.uniform(size=n_genes) - 1)
    DE_effects[~G_DE] = 0.0
    params_de = pd.DataFrame(
        {
            "id": [f"g{i + 1}" for i in np.where(G_DE)[0]],
            "DE_effect": DE_effects[G_DE],
        }
    )

    params: pd.DataFrame = pd.merge(
        params, params_de, on="id", how="outer", validate="one_to_one"
    )

    lib_size_fct: NDArray = rng.uniform(lib_size_var[0], lib_size_var[1], 2 * N)
    lib_size: NDArray = lib_size_fct * depth

    draw: NDArray = rng.choice(emp_dist.shape[0], n_genes, replace=True)

    size: NDArray = emp_dist["size"].values[draw].reshape(-1, 1)

    lambda_: NDArray = emp_dist["mu"].values[draw].reshape(-1, 1)

    DE_effects: NDArray = np.hstack([np.ones((n_genes, 1)), DE_effects.reshape(-1, 1)])

    lambda_: NDArray = np.repeat(lambda_, 2 * N, axis=1) * 2 ** (
        DE_effects[:, np.repeat([0, 1], N)] + A[:, np.repeat([0, 1], N)] * t_pattern
    )

    mu: NDArray = lambda_ / (np.sum(lambda_, axis=0) / lib_size)

    counts: NDArray = nbinom.rvs(n=size, p=size / (mu + size), random_state=seed)

    exp_design: pd.DataFrame = pd.DataFrame(
        {"time": np.tile(t, 2), "group": np.repeat(groups, N)},
        index=pd.Index(
            "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), 5))
            for _ in range(2 * N)
        ),
    )

    return {
        "counts": pd.DataFrame(
            counts,
            index=pd.Index([f"g{i + 1}" for i in range(n_genes)]),
            columns=exp_design.index,
        ),
        "params": params,
        "exp_design": exp_design,
    }
