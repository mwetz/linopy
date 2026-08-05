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


def simple_model(n_time: int = 12, infeasible: bool = False) -> linopy.Model:
    """Cap (no time dim -> root) + gen (time dim -> leaves), coupled by gen <= cap."""
    time = pd.Index(range(n_time), name="snapshot")
    m = linopy.Model()
    cap = m.add_variables(lower=0, name="cap")
    gen = m.add_variables(
        lower=0, upper=0 if infeasible else np.inf, coords=[time], name="gen"
    )
    m.add_constraints(gen <= cap, name="caplimit")
    m.add_constraints(
        gen >= xr.DataArray(np.linspace(1, 2, n_time), coords=[time]), name="demand"
    )
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


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_duals_and_runtime_are_returned() -> None:
    ref = simple_model(12)
    ref.solve("highs")

    m = simple_model(12)
    m.solve(
        "pipsipmpp", n_blocks=3, LINEAR_LEAF_SOLVER="mumps", LINEAR_ROOT_SOLVER="mumps"
    )

    for name in ("caplimit", "demand"):
        dual = m.constraints[name].dual.values
        assert np.isfinite(dual).all(), f"{name} duals must not be NaN"
        np.testing.assert_allclose(dual, ref.constraints[name].dual.values, atol=1e-5)

    # runtime is measured inside PIPS-IPM++ and reported through the solver report
    assert m.solver is not None
    assert m.solver.report is not None
    assert m.solver.report.runtime > 0.0


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_termination_status_map_covers_every_pips_status() -> None:
    """Every PIPS-IPM++ status must map to a linopy termination condition."""
    from pipsipmpppy import TerminationStatus

    mapping = PIPSIPMpp(model=None)._CONDITION_MAP
    missing = set(TerminationStatus) - set(mapping)
    assert not missing, (
        f"unmapped PIPS-IPM++ statuses: {sorted(s.name for s in missing)}"
    )


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
@pytest.mark.parametrize(
    "kwargs,options,expected",
    [
        ({}, {}, "optimal"),
        ({"infeasible": True}, {}, "infeasible"),
        ({}, {"STOP_AFTER_PRESOLVE": True}, "user_interrupt"),
    ],
)
def test_statuses_are_propagated(kwargs, options, expected) -> None:
    m = simple_model(12, **kwargs)
    _, condition = m.solve(
        "pipsipmpp",
        n_blocks=3,
        LINEAR_LEAF_SOLVER="mumps",
        LINEAR_ROOT_SOLVER="mumps",
        **options,
    )
    assert condition == expected
    # statuses without an iterate must not fabricate a solution
    if expected != "optimal":
        assert m.solver is not None
        assert m.solver.status.legacy_status.split(":")[0] != "SUCCESSFUL_TERMINATION"


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
@pytest.mark.parametrize("layout", ["monolithic", "distributed"])
def test_write_parquet_exports_without_solving(tmp_path, layout) -> None:
    """The annotated model can be written out for a later or remote solve."""
    pytest.importorskip("pyarrow")
    import pipsipmpppy

    m = simple_model(12)
    stem = PIPSIPMpp.write_parquet(m, tmp_path / "model", layout=layout, n_blocks=3)

    manifest = pipsipmpppy.read_manifest(stem)
    assert manifest["layout"] == layout
    assert manifest["n_blocks"] == 4  # the root counts alongside the three leaves
    assert manifest["n_cols"] == m.matrices.vlabels.size
    # the leaves carry the dispatch variables, the root the single capacity
    assert [block["n"] for block in manifest["blocks"]] == [1, 4, 4, 4]


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_write_parquet_records_a_maximisation(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    import pipsipmpppy

    m = simple_model(12)
    m.objective = -(10 * m.variables["cap"] + m.variables["gen"].sum())
    m.objective.sense = "max"

    stem = PIPSIPMpp.write_parquet(m, tmp_path / "model", n_blocks=3)
    # the problem holds minimisation costs; objcoef records how the model stated them
    assert pipsipmpppy.read_manifest(stem)["objcoef"] == -1.0


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_both_layouts_describe_the_same_problem(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    import pipsipmpppy

    m = simple_model(12)
    whole = pipsipmpppy.read_manifest(
        PIPSIPMpp.write_parquet(m, tmp_path / "whole", n_blocks=3)
    )
    split = pipsipmpppy.read_manifest(
        PIPSIPMpp.write_parquet(m, tmp_path / "split", layout="distributed", n_blocks=3)
    )

    assert (whole["n_rows"], whole["n_cols"]) == (split["n_rows"], split["n_cols"])
    keys = ("n", "my", "mz", "myl", "mzl")
    assert [[b[k] for k in keys] for b in whole["blocks"]] == [
        [b[k] for k in keys] for b in split["blocks"]
    ]


@pytest.fixture
def options_file(tmp_path):
    """A settings file, as a repeatable run would keep on disk."""
    path = tmp_path / "base.opt"
    path.write_text(
        "# base configuration\n"
        "SCALER                       geometricmean\n"
        "PRESOLVE_BOUND_STR_MAX_ITER  7\n"
    )
    return path


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_options_file_is_used_as_the_base(options_file) -> None:
    m = simple_model(9)
    _, condition = m.solve("pipsipmpp", n_blocks=3, options_file=str(options_file))
    assert condition == "optimal"


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_solver_options_override_the_file(options_file) -> None:
    """The file is the base; options passed alongside it win."""
    m = simple_model(9)
    _, condition = m.solve(
        "pipsipmpp",
        n_blocks=3,
        options_file=str(options_file),
        SCALER="none",
        PRESOLVE_BOUND_STR_MAX_ITER=2,
    )
    assert condition == "optimal"

    # both settings only steer how the solve is carried out, so the same optimum
    # comes back, to the solver's convergence tolerance
    plain = simple_model(9)
    plain.solve("pipsipmpp", n_blocks=3)
    assert m.objective.value == pytest.approx(plain.objective.value, abs=1e-5)


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_a_missing_options_file_is_reported(tmp_path) -> None:
    m = simple_model(9)
    with pytest.raises(FileNotFoundError, match="options file"):
        m.solve("pipsipmpp", n_blocks=3, options_file=str(tmp_path / "absent.opt"))


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_options_file_is_not_forwarded_as_a_solver_option() -> None:
    assert "options_file" in PIPSIPMpp._INTERFACE_OPTIONS
