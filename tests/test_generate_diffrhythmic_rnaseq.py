"""Tests for `generate_diffrhythmic_rnaseq`."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from pyfauxseq import generate_diffrhythmic_rnaseq

settings.register_profile("fast", max_examples=20)


@st.composite
def ordered_tuple(draw):
    n1 = draw(st.floats(min_value=0.25, max_value=1.0))
    n2 = draw(st.floats(min_value=n1, max_value=2.0))
    return (n1, n2)


@given(st.integers(min_value=2, max_value=20000))
def test_n_genes(n_genes):
    data = generate_diffrhythmic_rnaseq(n_genes=n_genes)
    assert data["counts"].shape[0] == n_genes


@given(
    st.integers(min_value=1, max_value=4),
    st.lists(st.floats(min_value=-100, max_value=100), min_size=4, max_size=8),
)
def test_n_samples(reps, t):
    data = generate_diffrhythmic_rnaseq(reps=reps, t=t)
    assert data["counts"].shape[1] == len(t) * reps * 2
    assert data["exp_design"].shape[0] == len(t) * reps * 2


@given(
    st.integers(min_value=2, max_value=20000),
    st.integers(min_value=1, max_value=4),
    st.lists(st.floats(min_value=-100, max_value=100), min_size=4, max_size=8),
)
def test_count_values(n_genes, reps, t):
    data = generate_diffrhythmic_rnaseq(n_genes=n_genes, reps=reps, t=t)
    assert all(data["counts"].dtypes == np.int64)
    assert (data["counts"].to_numpy() >= 0).all()


@given(
    st.integers(min_value=1, max_value=4),
    st.lists(st.floats(min_value=-100, max_value=100), min_size=4, max_size=8),
)
def test_unique_colnames(reps, t):
    data = generate_diffrhythmic_rnaseq(reps=reps, t=t)
    assert data["counts"].columns.is_unique
    assert sorted(data["exp_design"].index.to_list()) == sorted(
        data["counts"].columns.to_list()
    )


@given(
    st.integers(min_value=1000, max_value=20000),  # n_genes
    st.floats(min_value=0.01, max_value=1.0),  # rhy_frac
    st.floats(min_value=0.0, max_value=2.0),  # min_A_effect
    st.floats(min_value=0.0, max_value=2.0),  # min_DE_effect
)
def test_params(n_genes, rhy_frac, min_A_effect, min_DE_effect):
    data = generate_diffrhythmic_rnaseq(
        n_genes=n_genes,
        rhy_frac=rhy_frac,
        min_A_effect=min_A_effect,
        min_DE_effect=min_DE_effect,
    )
    params = data["params"]
    assert params["category"].isin(["loss", "gain", "change", "same", np.nan]).all()
    assert params["id"].isin(data["counts"].index.to_list()).all()

    assert (params["DE_effect"].dropna().abs() >= np.log2(1 + min_DE_effect)).all()

    assert (params["phi_ctrl"].dropna() >= 0).all() and (
        params["phi_ctrl"].dropna() <= 2 * np.pi
    ).all()
    assert (params["phi_expt"].dropna() >= 0).all() and (
        params["phi_expt"].dropna() <= 2 * np.pi
    ).all()

    assert (
        params[params["category"] == "loss"]["A_ctrl"] >= np.log2(1 + min_A_effect)
    ).all() and (params[params["category"] == "loss"]["A_expt"] == 0.0).all()
    assert (
        params[params["category"] == "gain"]["A_expt"] >= np.log2(1 + min_A_effect)
    ).all() and (params[params["category"] == "gain"]["A_ctrl"] == 0.0).all()
    assert (
        params[params["category"] == "change"]["A_expt"] >= np.log2(1 + min_A_effect)
    ).all() and (
        params[params["category"] == "change"]["A_ctrl"] >= np.log2(1 + min_A_effect)
    ).all()
    assert (
        params[params["category"] == "same"]["A_ctrl"]
        == params[params["category"] == "same"]["A_expt"]
    ).all()

    assert (
        not np.logical_xor(params["category"].isna(), params["A_ctrl"].isna()).any()
    )

@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_seeding(seed):
    data_1 = generate_diffrhythmic_rnaseq(seed=seed)
    data_2 = generate_diffrhythmic_rnaseq(seed=seed)

    assert (
        data_1["counts"].equals(data_2["counts"])
        and data_1["params"].equals(data_2["params"])
        and data_1["exp_design"].equals(data_2["exp_design"])
    )
