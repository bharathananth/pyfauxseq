from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy.optimize import minimize
from .utils import downsample

def nb_fit(x):
    if np.var(x) < np.mean(x):
        return pd.Series({'size': 1e6, 'mu': np.mean(x)})

    try:
        m = np.mean(x)
        v = np.var(x)

        def neg_log_likelihood(params):
            size, mu = params
            return -np.sum(nbinom.logpmf(x, n=size, p=size / (size + mu)))

        initial_guess = [m / (v - m), m]
        bounds = [(0, 1e6), (0, 1e6)]
        result = minimize(neg_log_likelihood, initial_guess, bounds=bounds, method='L-BFGS-B')

        if result.success:
            return pd.Series({'size': result.x[0], 'mu': m})
        else:
            return pd.Series({'size': np.nan, 'mu': np.nan})
    except Exception:
        return pd.Series({'size': np.nan, 'mu': np.nan})

def estimate_disp_dist(counts, parallel=True, ncores=None):
    counts = downsample(counts, parallel=parallel, ncores=ncores)

    if parallel:
        if ncores is None:
            ncores = min(32, (os.cpu_count() or 1) + 4)  # Default to number of CPUs
        with ThreadPoolExecutor(max_workers=ncores) as executor:
            futures = [executor.submit(nb_fit, counts[i, :]) for i in range(counts.shape[0])]
            ests = [future.result() for future in as_completed(futures)]
    else:
        ests = [nb_fit(counts[i, :]) for i in range(counts.shape[0])]

    final_ests = pd.DataFrame(ests).dropna()
    return final_ests

# Example usage
if __name__ == "__main__":
    counts = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    disp_dist = estimate_disp_dist(counts, parallel=True, ncores=2)
    print(disp_dist)
