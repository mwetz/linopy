"""
Tests for the PIPS-IPM++ solver interface and its block derivation.

The block-derivation tests run without PIPS-IPM++ installed; the solve tests are
skipped unless ``pipsipmpppy`` is importable.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy
from linopy.solvers import PIPSIPMpp

pipsipmpp_available = PIPSIPMpp.is_available()


def simple_model(n_time: int = 12) -> linopy.Model:
    """Cap (no time dim -> root) + gen (time dim -> leaves), coupled by gen <= cap."""
    time = pd.Index(range(n_time), name="snapshot")
    m = linopy.Model()
    cap = m.add_variables(lower=0, name="cap")
    gen = m.add_variables(lower=0, coords=[time], name="gen")
    m.add_constraints(gen <= cap, name="caplimit")
    m.add_constraints(gen >= xr.DataArray(np.linspace(1, 2, n_time), coords=[time]))
    m.objective = 10 * cap + gen.sum()
    return m


def test_blocks_from_dim_is_contiguous_and_one_based() -> None:
    m = simple_model(12)
    blocks = PIPSIPMpp._blocks_from_dim(m, "snapshot", 4)

    assert blocks.dims == ("snapshot",)
    # leaves are 1..N; 0 is reserved for the root/coupling block
    assert blocks.values.tolist() == [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3


def test_blocks_from_dim_uneven_split() -> None:
    blocks = PIPSIPMpp._blocks_from_dim(simple_model(10), "snapshot", 3)
    counts = np.bincount(blocks.values)[1:]
    assert counts.sum() == 10
    assert counts.max() - counts.min() <= 1


def test_blocks_from_dim_rejects_unknown_dim() -> None:
    with pytest.raises(ValueError, match="not found in the model variables"):
        PIPSIPMpp._blocks_from_dim(simple_model(), "does-not-exist", 2)


@pytest.mark.parametrize("n_blocks", [0, 13])
def test_blocks_from_dim_rejects_bad_count(n_blocks: int) -> None:
    with pytest.raises(ValueError, match="must be between"):
        PIPSIPMpp._blocks_from_dim(simple_model(12), "snapshot", n_blocks)


def test_variable_blocks_put_dimensionless_variables_in_the_root() -> None:
    m = simple_model(12)
    blocks = PIPSIPMpp._blocks_from_dim(m, "snapshot", 4)
    label_block, n_leaves = PIPSIPMpp._variable_blocks(m, blocks)

    assert n_leaves == 4
    # `cap` has no snapshot dimension -> coupling variable in the root block
    assert label_block[m.variables["cap"].labels.values.ravel()].tolist() == [0]
    # `gen` is spread over the leaves in time order
    gen_blocks = label_block[m.variables["gen"].labels.values.ravel()]
    assert gen_blocks.tolist() == [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_missing_blocks_raises_a_helpful_error() -> None:
    m = simple_model()
    with pytest.raises(ValueError, match="needs a block structure"):
        m.solve("pipsipmpp")


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_pipsipmpp_matches_highs() -> None:
    ref = simple_model(12)
    ref.solve("highs")

    m = simple_model(12)
    m.solve(
        "pipsipmpp", n_blocks=3, LINEAR_LEAF_SOLVER="mumps", LINEAR_ROOT_SOLVER="mumps"
    )

    assert m.status == "ok"
    assert m.objective.value == pytest.approx(ref.objective.value, rel=1e-5)
    assert m.variables["cap"].solution.item() == pytest.approx(
        ref.variables["cap"].solution.item(), rel=1e-5
    )


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_explicit_model_blocks_are_used() -> None:
    """`model.blocks` set by hand takes precedence over the n_blocks option."""
    m = simple_model(12)
    m.blocks = xr.DataArray(np.repeat([1, 2], 6), dims=["snapshot"])
    m.solve("pipsipmpp", LINEAR_LEAF_SOLVER="mumps", LINEAR_ROOT_SOLVER="mumps")
    assert m.status == "ok"
