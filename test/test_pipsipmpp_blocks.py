"""
Tests for the PIPS-IPM++ solver interface and its block derivation.

The block-derivation tests run without PIPS-IPM++ installed; the solve tests are
skipped unless ``pipsipmpppy`` is importable.
"""

import importlib.util

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


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
@pytest.mark.parametrize("layout", ["monolithic", "distributed"])
def test_write_parquet_carries_model_names(tmp_path, layout) -> None:
    """Names make a plotted matrix readable, so they must match the model's own."""
    pytest.importorskip("pyarrow")
    import pipsipmpppy

    m = simple_model(6)
    stem = PIPSIPMpp.write_parquet(
        m, tmp_path / "named", layout=layout, n_blocks=3, names=True
    )
    names = pipsipmpppy.read_names(stem)

    assert names["cols"] == ["cap"] + [f"gen[{i}]" for i in range(6)]
    # rows are stored equalities first, and this model has only inequalities
    assert sorted(names["rows"]) == sorted(
        [f"caplimit[{i}]" for i in range(6)] + [f"demand[{i}]" for i in range(6)]
    )


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_names_line_up_with_the_rows_they_label(tmp_path) -> None:
    """A name on the wrong row would mislabel a plot without failing anywhere."""
    pq = pytest.importorskip("pyarrow.parquet")

    m = simple_model(6)
    stem = PIPSIPMpp.write_parquet(m, tmp_path / "named", n_blocks=3, names=True)
    cols = pq.read_table(f"{stem}.cols.parquet").to_pandas()
    rows = pq.read_table(f"{stem}.rows.parquet").to_pandas()
    amat = pq.read_table(f"{stem}.amat.parquet").to_pandas()

    # caplimit[i] is `gen[i] <= cap`, so it touches exactly cap and gen[i]
    row = rows.index[rows["name"] == "caplimit[3]"][0]
    touched = {cols["name"][c] for c in amat.loc[amat["row"] == row, "col"]}
    assert touched == {"cap", "gen[3]"}

    # cap has no block dimension, so it is the root/coupling variable
    assert int(cols.loc[cols["name"] == "cap", "partition"].iloc[0]) == 1
    assert int(cols.loc[cols["name"] == "gen[3]", "partition"].iloc[0]) != 1


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_names_can_be_left_out(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    import pipsipmpppy

    m = simple_model(6)
    plain = PIPSIPMpp.write_parquet(m, tmp_path / "plain", n_blocks=3, names=False)
    assert pipsipmpppy.read_names(plain) == {}
    # leaving the names out must not change the structure
    named = PIPSIPMpp.write_parquet(m, tmp_path / "named", n_blocks=3, names=True)
    keys = ("n", "my", "mz", "myl", "mzl")
    assert [
        [b[k] for k in keys] for b in pipsipmpppy.read_manifest(plain)["blocks"]
    ] == [[b[k] for k in keys] for b in pipsipmpppy.read_manifest(named)["blocks"]]


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_names_are_off_by_default(tmp_path) -> None:
    """Producing names costs a lookup per label, so they are opt-in."""
    pytest.importorskip("pyarrow")
    import pipsipmpppy

    m = simple_model(6)
    default = PIPSIPMpp.write_parquet(m, tmp_path / "default", n_blocks=3)
    assert pipsipmpppy.read_names(default) == {}


def _solution_beside(stem, objective: float = 12.5) -> None:
    """Write a solution whose values are the global index they belong to."""
    import pipsipmpppy
    from pipsipmpppy.flat import solver_order, write_solution

    order = solver_order(stem)
    write_solution(
        stem,
        order=order,
        status=pipsipmpppy.TerminationStatus.SUCCESSFUL_TERMINATION,
        objective=objective,
        runtime=0.0,
        iterations=7,
        primal=[float(c) for c in order.cols],
        dual_eq=[float(r) for r in order.eq_rows],
        dual_ineq=[float(r) for r in order.ineq_rows],
    )


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_read_parquet_solution_lands_on_the_model(tmp_path) -> None:
    pytest.importorskip("pyarrow")

    m = simple_model(6)
    stem = PIPSIPMpp.write_parquet(m, tmp_path / "model", n_blocks=3)
    _solution_beside(stem)

    status, condition = m.assign_result(PIPSIPMpp.read_parquet_solution(m, stem))

    assert (status, condition) == ("ok", "optimal")
    assert m.objective.value == 12.5
    # cap is written first, then the six gen entries, each carrying its position
    assert float(m.variables["cap"].solution) == 0.0
    assert list(m.variables["gen"].solution.values) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_read_parquet_solution_places_the_duals_by_row(tmp_path) -> None:
    pytest.importorskip("pyarrow")

    m = simple_model(6)
    stem = PIPSIPMpp.write_parquet(m, tmp_path / "model", n_blocks=3)
    _solution_beside(stem)
    m.assign_result(PIPSIPMpp.read_parquet_solution(m, stem))

    # every constraint here is an inequality, so the duals follow the row order
    duals = np.concatenate(
        [m.constraints["caplimit"].dual.values, m.constraints["demand"].dual.values]
    )
    assert sorted(duals) == list(range(12))


@pytest.mark.skipif(not pipsipmpp_available, reason="pipsipmpppy not installed")
def test_read_parquet_solution_flips_a_maximisation_back(tmp_path) -> None:
    pytest.importorskip("pyarrow")

    m = simple_model(6)
    m.objective.sense = "max"
    stem = PIPSIPMpp.write_parquet(m, tmp_path / "model", n_blocks=3)
    # the file holds the minimisation objective PIPS-IPM++ worked with
    _solution_beside(stem, objective=-12.5)

    m.assign_result(PIPSIPMpp.read_parquet_solution(m, stem))

    assert m.objective.value == 12.5
    assert float(m.constraints["caplimit"].dual[0]) == -0.0


# Automatic detection of block structures requires pipstools dependency
pipstools_available = importlib.util.find_spec("pipstools") is not None
mtkahypar_available = importlib.util.find_spec("mtkahypar") is not None

# linopy names a variable "gen[wind, 17]", so this captures the last coordinate;
# "cap" carries none and stays in the root
LAST_COORD = r"(\d+)\]$"

needs_pipstools = pytest.mark.skipif(
    not (pipsipmpp_available and pipstools_available),
    reason="pipsipmpppy and pipstools are both needed to derive a structure",
)


def annotation_kwargs(method: str) -> dict:
    if method == "regex":
        return {"annotation": "regex", "regex": LAST_COORD}
    return {"annotation": "hypergraph"}


def skip_if_unavailable(method: str) -> None:
    if method == "hypergraph" and not mtkahypar_available:
        pytest.skip("mtkahypar is needed for the hypergraph method")


@needs_pipstools
@pytest.mark.parametrize("method", ["regex", "hypergraph"])
def test_annotation_derives_a_structure_for_a_model_without_one(method: str) -> None:
    skip_if_unavailable(method)
    m = simple_model(12)
    assert m.blocks is None

    problem, _is_eq = PIPSIPMpp._structured_problem(
        m, n_blocks=4, **annotation_kwargs(method)
    )

    assert problem.n_blocks == 4
    assert np.bincount(problem.var_block, minlength=5)[1:].min() > 0
    problem.validate()


@needs_pipstools
def test_regex_keeps_the_coupling_variable_in_the_root() -> None:
    """`cap` matches no coordinate, so it belongs where every block can see it."""
    m = simple_model(12)
    problem, _is_eq = PIPSIPMpp._structured_problem(
        m, n_blocks=4, **annotation_kwargs("regex")
    )
    # cap is the first label, and the twelve gen entries follow
    assert problem.var_block[0] == 0
    assert (problem.var_block[1:] != 0).all()


@needs_pipstools
def test_deriving_without_a_block_count_is_refused() -> None:
    with pytest.raises(ValueError, match="n_blocks"):
        PIPSIPMpp._structured_problem(simple_model(12), annotation="hypergraph")


@needs_pipstools
def test_stating_no_structure_and_deriving_none_is_refused() -> None:
    """Without an annotation the old error still names all three ways out."""
    with pytest.raises(ValueError, match="annotation="):
        PIPSIPMpp._structured_problem(simple_model(12))


@needs_pipstools
@pytest.mark.parametrize("method", ["regex", "hypergraph"])
def test_a_derived_structure_solves_to_the_same_optimum(method: str) -> None:
    skip_if_unavailable(method)
    reference = simple_model(12)
    reference.solve("highs")

    m = simple_model(12)
    m.solve("pipsipmpp", n_blocks=4, **annotation_kwargs(method))

    assert m.status == "ok"
    assert m.objective.value == pytest.approx(reference.objective.value, abs=1e-6)
    assert np.allclose(
        m.variables["gen"].solution.values,
        reference.variables["gen"].solution.values,
        atol=1e-6,
    )


@needs_pipstools
@pytest.mark.parametrize("method", ["regex", "hypergraph"])
def test_a_derived_structure_survives_the_parquet_roundtrip(tmp_path, method) -> None:
    """Build here, solve there, read the answer back: the three steps apart."""
    skip_if_unavailable(method)
    pytest.importorskip("pyarrow")
    import pipsipmpppy
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() != 1:
        pytest.skip("this test drives MPI itself; run it on a single rank")

    kwargs = annotation_kwargs(method)

    in_memory = simple_model(12)
    in_memory.solve("pipsipmpp", n_blocks=4, **kwargs)

    # step one: write the annotated problem out, with no solver involved
    m = simple_model(12)
    stem = PIPSIPMpp.write_parquet(m, tmp_path / "model", n_blocks=4, **kwargs)
    # step two: solve from the files alone
    pipsipmpppy.solve_dataset(stem, comm, write_solution=True)
    # step three: read it back onto the model, which needs no solver either
    status, condition = m.assign_result(PIPSIPMpp.read_parquet_solution(m, stem))

    assert (status, condition) == ("ok", "optimal")
    assert m.objective.value == pytest.approx(in_memory.objective.value, abs=1e-6)
    assert np.allclose(
        m.variables["gen"].solution.values,
        in_memory.variables["gen"].solution.values,
        atol=1e-6,
    )
    assert np.allclose(
        m.constraints["demand"].dual.values,
        in_memory.constraints["demand"].dual.values,
        atol=1e-6,
    )
