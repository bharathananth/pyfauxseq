import numpy as np
import pandas as pd
from scipy.stats import nbinom
from .utils import load_dataset

def generate_rhy_rnaseq(t=np.arange(0, 21, 4),
                        reps=None,
                        period=24,
                        n_genes=10000,
                        rhy_frac=0.1,
                        min_A_effect=0.0,
                        A_spread=1,
                        emp_dist=None,
                        depth=1e7,
                        lib_size_var=(0.8, 1.2),
                        seed=None):
    
    rng = np.random.default_rng(seed)
    
    if reps is None:
        reps = np.ones(len(t), dtype=int)
    else:
        if not isinstance(reps, int) and (not isinstance(reps, list) or len(reps) != len(t)):
            raise ValueError("Length of reps must be 1 or the same as length of t")

    if emp_dist is None:
        emp_dist = load_dataset("Mm_liver_LD_NC.csv.gz")

    n_rhy = int(np.round(rhy_frac * n_genes))

    t = np.repeat(t, reps)
    N = len(t)

    G_rhy = rng.uniform(size=n_genes) <= rhy_frac

    A = np.ones((n_genes, )) + min_A_effect + \
                rng.exponential(1/A_spread, n_genes)
    A[~G_rhy] = 1.0
    phi = rng.uniform(size=n_genes) * 2 * np.pi * G_rhy
    params = pd.DataFrame({
        'id': [f"g{i+1}" for i in np.where(G_rhy)[0]] ,
        'A': np.log2(A[G_rhy]),
        'phi': phi[G_rhy]
    })

    t_pattern = np.cos(phi)[:, None] @ np.cos(2 * np.pi * t[None, :] / period) + \
                np.sin(phi)[:, None] @ np.sin(2 * np.pi * t[None, :] / period)

    lib_size_fct = rng.uniform(lib_size_var[0], lib_size_var[1], N)
    lib_size = lib_size_fct * depth

    draw = rng.choice(emp_dist.shape[0], n_genes, replace=True)
    lambda_ = emp_dist['mu'].values[draw].reshape(-1, 1)

    size = emp_dist['size'].values[draw].reshape(-1, 1)
    
    lambda_ = np.repeat(lambda_, N, axis=1) * A[:, None] ** t_pattern

    mu = lambda_ / (np.sum(lambda_, axis=0) / lib_size)

    counts = nbinom.rvs(n = size, p = size / (mu + size))

    exp_design = pd.DataFrame({
        'time': t,
    }, 
    index=[''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'), 5)) 
            for _ in range(N)])

    return {
        'counts': pd.DataFrame(counts, index=[f"g{i+1}" for i in range(n_genes)], 
                                columns=exp_design.index),
        'params': params,
        'exp_design': exp_design
    }

def generate_diffrhy_rnaseq(t=np.arange(0, 21, 4),
                            reps=None,
                            period=24,
                            n_genes=10000,
                            rhy_frac=0.1,
                            min_A_effect=0.0,
                            A_spread=1,
                            DE_frac=0.1,
                            min_DE_effect=0.5,
                            DE_spread=1,
                            groups=("ctrl", "expt"),
                            emp_dist=None,
                            depth=1e7,
                            lib_size_var=(0.8, 1.2),
                            seed=None):
    
    rng = np.random.default_rng(seed)
    
    if reps is None:
        reps = np.ones(len(t), dtype=int)
    else:
        if not isinstance(reps, int) and (not isinstance(reps, list) or len(reps) != len(t)):
            raise ValueError("Length of reps must be 1 or the same as length of t")

    if emp_dist is None:
        emp_dist = load_dataset("Mm_liver_LD_NC.csv.gz")

    t = np.repeat(t, reps)
    N = len(t)

    G_rhy = rng.uniform(size=n_genes) <= rhy_frac
    DR_groups = rng.choice(["gain", "loss", "change", "same"], 
                           sum(G_rhy), replace=True)
    A = np.ones((n_genes, 2)) + min_A_effect + \
                rng.exponential(1/A_spread, (n_genes, 2))
    A[~G_rhy, :] = 1.0
    phi = rng.uniform(size=(n_genes, 2)) * 2 * np.pi * G_rhy.reshape(-1, 1)
    print(A[~G_rhy,:][1:10,])

    G_rhy_index = np.where(G_rhy)[0]
    for i in range(len(DR_groups)):
        if DR_groups[i] == "gain":
            A[G_rhy_index[i], 0] = 1.0
        elif DR_groups[i] == "loss":
            A[G_rhy_index[i], 1] = 1.0
        elif DR_groups[i] == "same":
            A[G_rhy_index[i], 0] = A[G_rhy_index[i], 1]
            phi[G_rhy_index[i], 1] = phi[G_rhy_index[i], 0]

    params = pd.DataFrame({
        'id': [f"g{i+1}" for i in np.where(G_rhy)[0]],
        'category': DR_groups,
        'A_1': np.log2(A[G_rhy, 0]),
        'A_2': np.log2(A[G_rhy, 1]),
        'phi_1': phi[G_rhy, 0],
        'phi_2': phi[G_rhy, 1]
    })
    params.columns = ['id', 'category', 'A_ctrl', 'A_expt', 
                      'phi_ctrl', 'phi_expt']

    t_pattern = np.hstack([
        np.cos(phi[:, [0]]) @ np.cos(2 * np.pi * t[None, :] / period) + \
        np.sin(phi[:, [0]]) @ np.sin(2 * np.pi * t[None, :] / period),
        np.cos(phi[:, [1]]) @ np.cos(2 * np.pi * t[None, :] / period) + \
        np.sin(phi[:, [1]]) @ np.sin(2 * np.pi * t[None, :] / period)
    ])
    
    print(t_pattern.shape)

    G_DE = rng.uniform(size=n_genes) <= DE_frac
    DE_effects = (1 + min_DE_effect + 
                    rng.exponential(1/DE_spread, n_genes)) ** \
                        np.sign(2 * rng.uniform(size=n_genes) - 1)
    DE_effects[~G_DE] = 1.0
    params_de = pd.DataFrame({
        'id': [f"g{i+1}" for i in np.where(G_DE)[0]],
        'DE_effect': np.log2(DE_effects[G_DE])
    })

    params = pd.merge(params, params_de, 
                      on='id', 
                      how='outer', 
                      validate="one_to_one")

    lib_size_fct = rng.uniform(lib_size_var[0], lib_size_var[1], 2 * N)
    lib_size = lib_size_fct * depth

    draw = rng.choice(emp_dist.shape[0], 
                      n_genes, 
                      replace=True)

    size = emp_dist['size'].values[draw].reshape(-1, 1)

    lambda_ = emp_dist['mu'].values[draw].reshape(-1, 1)

    DE_effects = np.hstack([np.ones((n_genes, 1)), DE_effects.reshape(-1, 1)])

    lambda_ = np.repeat(lambda_, 2 * N, axis=1) * \
                DE_effects[:, np.repeat([0, 1], N)] * \
                A[:, np.repeat([0, 1], N)] ** t_pattern

    mu = lambda_ / (np.sum(lambda_, axis=0) / lib_size)

    counts = nbinom.rvs(n = size, p = size / (mu + size))

    exp_design = pd.DataFrame({
        'time': np.tile(t, 2),
        'group': np.repeat(groups, N)
    }, index=[''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'), 5)) 
                                                        for _ in range(2 * N)])

    return {
        'counts': pd.DataFrame(counts, 
                               index=[f"g{i+1}" for i in range(n_genes)], 
                               columns=exp_design.index),
        'params': params,
        'exp_design': exp_design
    }