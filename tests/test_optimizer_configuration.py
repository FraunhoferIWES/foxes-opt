import ast
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import yaml
import foxes
from foxes.utils import Dict
from iwopy import SimpleObjective, SimpleProblem
from iwopy.core import MultiObjOptResults
from scipy.sparse import isspmatrix_csr
from xarray import open_dataset

from foxes_opt.input.yaml import dict as yaml_dict
from foxes_opt.pipelines.stages import LayoutOptimizerStage
from foxes_opt.output import MultiObjResultsWriter, SingleObjResultsWriter


class _Quadratic(SimpleObjective):
    def f(self, value):
        return value**2

    def g(self, var, value, components):
        return 2 * value


def _read_optimizer(monkeypatch, parameters):
    problem = SimpleProblem(
        "configuration",
        float_vars=["x"],
        init_values_float=[1.0],
        min_values_float=[-3.0],
        max_values_float=[3.0],
    )
    problem.add_objective(_Quadratic(problem))
    monkeypatch.setattr(
        yaml_dict, "foxes_read_dict", lambda *args, **kwargs: (None, None)
    )
    monkeypatch.setattr(yaml_dict.FarmOptProblem, "new", lambda **kwargs: problem)
    inputs = Dict(
        {
            "optimization": {
                "problem": {"problem_type": "synthetic", "objectives": []},
                "optimizer": parameters,
            }
        }
    )
    return yaml_dict.read_dict(inputs, verbosity=0)[2]


@pytest.mark.parametrize(
    "parameters",
    [
        {"family": "iwopy", "optimizer": "SLSQP", "max_iterations": 100},
        {"family": "scipy", "optimizer": "trust-constr", "max_iterations": 100},
        {
            "family": "pymoo",
            "optimizer": "GA",
            "pop_size": 4,
            "seed": 1,
            "termination": "iterations",
            "max_iterations": 2,
        },
    ],
)
def test_yaml_constructs_and_runs_current_optimizer_contract(monkeypatch, parameters):
    optimizer = _read_optimizer(monkeypatch, parameters)
    jacobian = optimizer.problem.get_jacobian(np.array([], dtype=int), np.array([1.0]))
    assert isspmatrix_csr(jacobian)
    np.testing.assert_allclose(jacobian.toarray(), [[2.0]])
    results = optimizer.solve(verbosity=0)
    assert np.all(np.isfinite(results.objs))
    optimizer.finalize(results, verbosity=0)


@pytest.mark.parametrize(
    ("parameters", "exception", "message"),
    [
        (
            {"family": "unknown", "optimizer": "GA"},
            ValueError,
            "Unknown optimizer family",
        ),
        (
            {"family": "iwopy", "optimizer": "SLSQP", "scipy_pars": {}},
            TypeError,
            "scipy_pars",
        ),
        (
            {"family": "iwopy", "optimizer": "SLSQP", "maxiter": 10},
            TypeError,
            "maxiter",
        ),
        (
            {"family": "iwopy", "optimizer": "SLSQP", "max_evaluations": 10},
            ValueError,
            "max_evaluations is not supported",
        ),
    ],
)
def test_yaml_rejects_invalid_optimizer_configuration(
    monkeypatch, parameters, exception, message
):
    with pytest.raises(exception, match=message):
        _read_optimizer(monkeypatch, parameters)


def test_public_yaml_example_uses_current_optimizer_contract(monkeypatch):
    path = Path(__file__).parents[1] / "examples/yaml_input/inputs.yaml"
    parameters = yaml.safe_load(path.read_text())["optimization"]["optimizer"]
    optimizer = _read_optimizer(monkeypatch, parameters)
    assert optimizer.optimizer == "GA"
    assert optimizer.vectorize is True
    assert optimizer.seed == 42


@pytest.mark.parametrize("nofig", [False, True])
def test_yaml_runs_constructor_outputs_and_preserves_phase_order(
    monkeypatch, tmp_path, nofig
):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda **kwargs: None)
    path = Path(__file__).parents[1] / "examples/yaml_input/inputs.yaml"
    inputs = Dict.from_yaml(path, verbosity=0)
    inputs["engine"] = {"engine_type": "numpy", "verbosity": 0}
    inputs["optimization"]["optimizer"].update(pop_size=4, max_iterations=2)

    results, outputs = yaml_dict.run_dict(
        inputs, output_dir=tmp_path, verbosity=0, nofig=nofig
    )

    assert len(outputs) == len(inputs["outputs"])
    assert all(values is not None for _, values in outputs)
    assert outputs[0][1][0].attrs["problem"] == results.pname
    with open_dataset(tmp_path / "results.nc") as written:
        np.testing.assert_allclose(written["vars_float"], results.vars_float)
    assert (tmp_path / "result.png").exists() is not nofig


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "Algorithm and optimization results are required"),
        ({"functions": [{"function": "write_nc"}]}, "constructor options"),
    ],
)
def test_yaml_optimization_outputs_reject_invalid_inputs(options, message):
    inputs = Dict({"outputs": [{"output_type": "SingleObjResultsWriter", **options}]})
    with pytest.raises(ValueError, match=message):
        yaml_dict.run_outputs(inputs, with_engine=True, verbosity=0)


@pytest.mark.parametrize("multi", [False, True])
def test_result_writers_use_parent_output_lifecycle(monkeypatch, tmp_path, multi):
    optimizer = _read_optimizer(
        monkeypatch, {"family": "iwopy", "optimizer": "SLSQP", "max_iterations": 5}
    )
    result = optimizer.solve(verbosity=0)
    if multi:
        problem = SimpleProblem(
            "multi_writer", float_vars=["x"], init_values_float=[1.0]
        )
        problem.add_objective(_Quadratic(problem, name="first"))
        problem.add_objective(_Quadratic(problem, name="second"))
        problem.initialize(verbosity=0)
        result = MultiObjOptResults(
            problem,
            np.array([True, False]),
            np.empty((2, 0), dtype=int),
            np.array([[1.0], [2.0]]),
            np.array([[1.0, 1.0], [4.0, 4.0]]),
            np.empty((2, 0)),
            None,
        )
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([0.0, 0.0], ["NREL5MW"]), verbosity=0)
    algo = foxes.algorithms.Downwind(
        farm,
        foxes.input.states.SingleStateStates(ws=9, wd=270, ti=0.08, rho=1.225),
        wake_models=[],
        verbosity=0,
    )

    def unexpected_science(*args, **kwargs):
        pytest.fail("Optimization writers must not calculate farm or point fields")

    monkeypatch.setattr(algo, "calc_farm", unexpected_science)
    monkeypatch.setattr(algo, "calc_points", unexpected_science)
    writer = MultiObjResultsWriter if multi else SingleObjResultsWriter
    output = writer(
        result, out_dir=tmp_path, to_file="solution.nc", write_pars={"verbosity": 0}
    )
    dataset = algo.calc_outputs(outputs=output)
    np.testing.assert_allclose(dataset["vars_float"], result.vars_float)
    assert "variables_int" not in dataset.dims
    assert "constraints" not in dataset.dims
    with open_dataset(tmp_path / "solution.nc") as written:
        np.testing.assert_allclose(written["vars_float"], result.vars_float)
        if multi:
            np.testing.assert_array_equal(written["success"], [1, 0])
    if multi:
        np.testing.assert_allclose(dataset["objs"], [[1.0, 1.0], [4.0, 4.0]])
        np.testing.assert_array_equal(dataset["success"], [1, 0])
        assert "success" not in dataset.attrs
    with pytest.raises(TypeError, match="Require results of type"):
        (SingleObjResultsWriter if multi else MultiObjResultsWriter)(result)


@pytest.mark.parametrize("label", ["$solution", "$", "solution", ["$solution"]])
def test_yaml_optimization_writer_result_labels(monkeypatch, tmp_path, label):
    path = Path(__file__).parents[1] / "examples/yaml_input/inputs.yaml"
    inputs = Dict.from_yaml(path, verbosity=0)
    inputs["engine"] = {"engine_type": "numpy", "verbosity": 0}
    inputs["optimization"]["optimizer"].update(pop_size=4, max_iterations=1)
    algo, _, optimizer = yaml_dict.read_dict(inputs, output_dir=tmp_path, verbosity=0)
    result = optimizer.solve(verbosity=0)
    definitions = Dict(
        {"outputs": [{"output_type": "SingleObjResultsWriter", "result_labels": label}]}
    )
    if label == "$solution":
        outputs, storage = yaml_dict.run_outputs(
            definitions, algo, opt_results=result, with_engine=True, verbosity=0
        )
        assert storage["$solution"] is outputs[0][1][0]
    else:
        with pytest.raises(ValueError, match="simple"):
            yaml_dict.run_outputs(
                definitions, algo, opt_results=result, with_engine=True, verbosity=0
            )


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("reserved", ["problem", "family"])
def test_pipeline_rejects_reserved_factory_parameters(repair, reserved, tmp_path):
    key = "boundary_repair_optimizer_pars" if repair else "optimizer_pars"
    stage = LayoutOptimizerStage(
        optimizer_family="iwopy", **{key: {reserved: "invalid"}}
    )
    pipeline = SimpleNamespace(
        farm_boundary=object(),
        states=object(),
        find_stage=lambda name: 0,
        base_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="reserved parameter"):
        stage.initialize(pipeline)


_ROOT = Path(__file__).parents[1]
_WORKFLOWS = sorted(_ROOT.glob("examples/**/*.py")) + sorted(
    _ROOT.glob("notebooks/*.ipynb")
)
_REMOVED_PARAMETERS = {
    "algo_pars",
    "problem_pars",
    "setup_pars",
    "term_pars",
    "scipy_pars",
    "vectorized",
    "maxiter",
    "max_iter",
    "gen",
    "n_max_gen",
    "n_max_evals",
    "budget",
    "popsize",
}


def _is_adapter_name(name):
    return name.startswith("Optimizer_") or name in {"GG", "SLSQP"}


def _factory_calls(source):
    source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
    calls = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("iwopy"):
            assert not any(_is_adapter_name(alias.name) for alias in node.names)
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else getattr(node.func, "attr", "")
        )
        assert not _is_adapter_name(name)
        if name == "create_optimizer":
            keywords = {keyword.arg for keyword in node.keywords}
            assert {"family", "optimizer", "problem"} <= keywords
            assert not _REMOVED_PARAMETERS & keywords
            calls.append(node)
    return calls


@pytest.mark.parametrize(
    "path", _WORKFLOWS, ids=lambda path: str(path.relative_to(_ROOT))
)
def test_public_workflows_use_only_optimizer_factory(path):
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text())["cells"]
        sources = [
            "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
        ]
    else:
        sources = [path.read_text()]
    calls = [call for source in sources for call in _factory_calls(source)]
    if path.suffix == ".ipynb" or path.stem in {
        "run_gg",
        "run_slsqp",
        "run_ipopt",
        "run_pymoo",
    }:
        assert calls
