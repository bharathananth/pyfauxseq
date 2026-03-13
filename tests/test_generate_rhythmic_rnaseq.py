"""Tests for `pyfauxseq` package."""
import numpy as np
import pytest as pt
from hypothesis import given, settings
from hypothesis import strategies as st

from pyfauxseq import generate_rhythmic_rnaseq

settings.register_profile("fast", max_examples=20)

@st.composite
def ordered_tuple(draw):
    n1 = draw(st.floats(min_value=0.25, max_value=1.0))
    n2 = draw(st.floats(min_value=n1, max_value=2.0))
    return (n1, n2)

@given(st.integers(min_value=2, max_value=20000))
def test_n_genes(n_genes):
    data = generate_rhythmic_rnaseq(n_genes=n_genes)
    assert data["counts"].shape[0] == n_genes

@given(st.integers(min_value=1, max_value=4),
       st.lists(st.floats(min_value=-100, max_value=100),
                min_size=4, max_size=8))
def test_n_samples(reps, t):
    data = generate_rhythmic_rnaseq(reps=reps, t=t)
    assert data["counts"].shape[1] == len(t) * reps
    assert data["exp_design"].shape[0] == len(t) * reps

@given(st.integers(min_value=2, max_value=20000),
       st.integers(min_value=1, max_value=4),
       st.lists(st.floats(min_value=-100, max_value=100),
                min_size=4, max_size=8))
def test_count_values(n_genes, reps, t):
    data = generate_rhythmic_rnaseq(n_genes=n_genes, reps=reps, t=t)
    assert all(data["counts"].dtypes == np.int64)
    assert (data["counts"].to_numpy()>=0).all()

@given(st.integers(min_value=1, max_value=4),
       st.lists(st.floats(min_value=-100, max_value=100),
                min_size=4, max_size=8))
def test_unique_colnames(reps, t):
    data = generate_rhythmic_rnaseq(reps=reps, t=t)
    assert data["counts"].columns.is_unique
    assert (sorted(data["exp_design"].index.to_list()) ==
                                    sorted(data["counts"].columns.to_list()))

@given(st.integers(min_value=1000, max_value=20000), # n_genes
       st.integers(min_value=1, max_value=4), # reps
       st.floats(min_value=0.01, max_value=1.0), # rhy_frac
       st.integers(min_value=2000, max_value=400000), # depth
       st.floats(min_value=0.0, max_value=2.0), # min_A_effect
       ordered_tuple(), # lib_size_var
       )
def test_params(n_genes, reps, rhy_frac, depth, min_A_effect, lib_size_var):
    data = generate_rhythmic_rnaseq(n_genes=n_genes, reps=reps, rhy_frac=rhy_frac,
                                    min_A_effect=min_A_effect, depth=depth,
                                    lib_size_var=lib_size_var)
    assert data["params"].shape[0]/n_genes == pt.approx(rhy_frac, abs=0.1)
    assert data["params"]["id"].isin(data["counts"].index.to_list()).all()
    assert (data["params"]["A"] >= np.log2(1 + min_A_effect)).all()
    assert (data["params"]["phi"] >=0).all() and \
                        (data["params"]["phi"] <= 2*np.pi).all()

@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_seeding(seed):
    data_1 = generate_rhythmic_rnaseq(seed=seed)
    data_2 = generate_rhythmic_rnaseq(seed=seed)

    assert (data_1["counts"].equals(data_2["counts"]) and
            data_1["params"].equals(data_2["params"]) and
            data_1["exp_design"].equals(data_2["exp_design"]))

