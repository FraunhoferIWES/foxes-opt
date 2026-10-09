import csv
from types import SimpleNamespace

import numpy as np
import pytest
from iwopy.core import OptimizerCallbackData

from foxes_opt.callbacks import WriteOptimizationHistoryCallback


class _Problem:
    n_objectives = 1
    maximize_objs = np.array([True])

    def check_constraints_individual(self, values):
        return values <= 0.0


class _InitialProblem(_Problem):
    def __init__(self, n_vars_int=0, n_vars_float=1):
        self.n_vars_int = n_vars_int
        self.n_vars_float = n_vars_float
        self.evaluations = []

    def initial_values_int(self):
        return np.full(self.n_vars_int, 3, dtype=int)

    def initial_values_float(self):
        return np.full(self.n_vars_float, 2.0)

    def evaluate_individual(self, variables_int, variables_float):
        self.evaluations.append((variables_int.copy(), variables_float.copy()))
        return np.array([4.0]), np.array([1.0, -1.0])


def _data(iteration, objectives, constraints):
    n_pop = 1 if objectives is None else len(objectives)
    return OptimizerCallbackData(
        event="iteration",
        iteration=iteration,
        vars_int=np.empty((n_pop, 0), dtype=int),
        vars_float=np.zeros((n_pop, 1)),
        objs=None if objectives is None else np.asarray(objectives)[:, None],
        cons=None if constraints is None else np.asarray(constraints),
    )


def _rows(file_path):
    with file_path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize("iteration_offset", [0, 7])
@pytest.mark.parametrize("n_vars_int,n_vars_float", [(0, 1), (1, 1), (1, 0), (0, 0)])
def test_history_records_initial_iteration(
    tmp_path, iteration_offset, n_vars_int, n_vars_float
):
    file_path = tmp_path / "history.csv"
    problem = _InitialProblem(n_vars_int, n_vars_float)
    callback = WriteOptimizationHistoryCallback(
        file_path, iteration_offset=iteration_offset, write_initial=True
    )
    callback.initialize(SimpleNamespace(problem=problem))

    assert len(problem.evaluations) == 1
    variables_int, variables_float = problem.evaluations[0]
    np.testing.assert_array_equal(variables_int, problem.initial_values_int())
    np.testing.assert_array_equal(variables_float, problem.initial_values_float())
    assert _rows(file_path) == [
        {
            "iteration": str(iteration_offset),
            "objective": "4.0",
            "n_violated_constraints": "1",
        }
    ]
    callback.notify(_data(1, [5.0], [[-1.0, -2.0]]))
    assert [row["iteration"] for row in _rows(file_path)] == [
        str(iteration_offset),
        str(iteration_offset + 1),
    ]


@pytest.mark.parametrize("write_initial", [None, False])
def test_history_initial_row_is_opt_in(tmp_path, write_initial):
    file_path = tmp_path / "history.csv"
    problem = _InitialProblem()
    options = {} if write_initial is None else {"write_initial": write_initial}
    callback = WriteOptimizationHistoryCallback(file_path, **options)
    callback.initialize(SimpleNamespace(problem=problem))

    assert problem.evaluations == []
    assert _rows(file_path) == []


def test_history_preserves_initial_integer_values(tmp_path, monkeypatch):
    problem = _InitialProblem(n_vars_int=1, n_vars_float=0)
    initial_values = np.array([2**40], dtype=np.int64)
    monkeypatch.setattr(problem, "initial_values_int", lambda: initial_values)
    callback = WriteOptimizationHistoryCallback(
        tmp_path / "history.csv", write_initial=True
    )
    callback.initialize(SimpleNamespace(problem=problem))

    np.testing.assert_array_equal(problem.evaluations[0][0], initial_values)


@pytest.mark.parametrize("write_initial", [None, 1, "true"])
def test_history_rejects_non_boolean_write_initial(tmp_path, write_initial):
    with pytest.raises(TypeError, match="write_initial must be a boolean"):
        WriteOptimizationHistoryCallback(
            tmp_path / "history.csv", write_initial=write_initial
        )


@pytest.mark.parametrize(
    "method,error",
    [("initial_values_int", "integer"), ("initial_values_float", "float")],
)
def test_history_requires_defined_initial_variables(
    tmp_path, monkeypatch, method, error
):
    file_path = tmp_path / "history.csv"
    problem = _InitialProblem(1, 1)
    monkeypatch.setattr(problem, method, lambda: None)
    callback = WriteOptimizationHistoryCallback(file_path, write_initial=True)

    with pytest.raises(ValueError, match=f"Initial {error} variables are undefined"):
        callback.initialize(SimpleNamespace(problem=problem))

    assert problem.evaluations == []
    assert _rows(file_path) == []


def test_history_reports_selected_initial_objective_without_constraints(
    tmp_path, monkeypatch
):
    file_path = tmp_path / "history.csv"
    problem = _InitialProblem()
    problem.n_objectives = 2
    problem.maximize_objs = np.array([True, False])
    monkeypatch.setattr(
        problem,
        "evaluate_individual",
        lambda variables_int, variables_float: (np.array([4.0, 9.0]), np.empty(0)),
    )
    callback = WriteOptimizationHistoryCallback(
        file_path, objective=1, write_initial=True
    )
    callback.initialize(SimpleNamespace(problem=problem))

    assert _rows(file_path) == [
        {"iteration": "0", "objective": "9.0", "n_violated_constraints": "0"}
    ]


def test_history_initial_row_preserves_existing_append_rows(tmp_path):
    file_path = tmp_path / "history.csv"
    optimizer = SimpleNamespace(problem=_InitialProblem())
    initial = WriteOptimizationHistoryCallback(file_path)
    initial.initialize(optimizer)
    initial.notify(_data(185, [1.0], [[-1.0]]))
    restarted = WriteOptimizationHistoryCallback(
        file_path, iteration_offset=188, append=True, write_initial=True
    )
    restarted.initialize(optimizer)

    assert _rows(file_path) == [
        {"iteration": "185", "objective": "1.0", "n_violated_constraints": "0"},
        {"iteration": "188", "objective": "4.0", "n_violated_constraints": "1"},
    ]


def test_history_writes_best_objective_and_violation_count(tmp_path):
    file_path = tmp_path / "output" / "history.csv"
    callback = WriteOptimizationHistoryCallback(file_path)
    optimizer = SimpleNamespace(problem=_Problem())
    callback.initialize(optimizer)

    callback.notify(_data(1, [1.0, 3.0], [[-1.0, -2.0], [1.0, -1.0]]))
    callback.notify(_data(2, [4.0], [[-1.0, -2.0]]))

    assert _rows(file_path) == [
        {"iteration": "1", "objective": "3.0", "n_violated_constraints": "1"},
        {"iteration": "2", "objective": "4.0", "n_violated_constraints": "0"},
    ]

    callback.initialize(optimizer)
    assert _rows(file_path) == []


def test_history_preserves_iteration_when_values_are_unavailable(tmp_path):
    file_path = tmp_path / "history.csv"
    callback = WriteOptimizationHistoryCallback(file_path)
    callback.initialize(SimpleNamespace(problem=_Problem()))

    callback.notify(_data(7, None, None))

    assert _rows(file_path) == [
        {"iteration": "7", "objective": "", "n_violated_constraints": ""}
    ]


def test_history_appends_with_restart_offset(tmp_path):
    file_path = tmp_path / "history.csv"
    optimizer = SimpleNamespace(problem=_Problem())
    initial = WriteOptimizationHistoryCallback(file_path)
    initial.initialize(optimizer)
    initial.notify(_data(188, [1.0], [[-1.0]]))
    restarted = WriteOptimizationHistoryCallback(
        file_path,
        iteration_offset=188,
        append=True,
    )
    restarted.initialize(optimizer)

    restarted.notify(_data(1, [2.0], [[-1.0]]))

    assert [row["iteration"] for row in _rows(file_path)] == ["188", "189"]


@pytest.mark.parametrize("iteration_offset", [True, 1.5])
def test_history_rejects_non_integer_iteration_offset(tmp_path, iteration_offset):
    with pytest.raises(TypeError, match="iteration_offset must be an integer"):
        WriteOptimizationHistoryCallback(
            tmp_path / "history.csv",
            iteration_offset=iteration_offset,
        )


def test_history_rejects_negative_iteration_offset(tmp_path):
    with pytest.raises(ValueError, match="iteration_offset must be non-negative"):
        WriteOptimizationHistoryCallback(
            tmp_path / "history.csv",
            iteration_offset=-1,
        )


def test_history_rejects_non_boolean_append(tmp_path):
    with pytest.raises(TypeError, match="append must be a boolean"):
        WriteOptimizationHistoryCallback(tmp_path / "history.csv", append=1)


def test_history_rejects_malformed_append_header(tmp_path):
    file_path = tmp_path / "history.csv"
    file_path.write_text("wrong,header\n", encoding="utf-8")
    callback = WriteOptimizationHistoryCallback(file_path, append=True)

    with pytest.raises(ValueError, match="Unexpected optimization history header"):
        callback.initialize(SimpleNamespace(problem=_Problem()))


def test_history_rejects_unknown_objective(tmp_path):
    callback = WriteOptimizationHistoryCallback(tmp_path / "history.csv", objective=1)

    with pytest.raises(IndexError, match="outside"):
        callback.initialize(SimpleNamespace(problem=_Problem()))


def test_history_requires_iteration(tmp_path):
    callback = WriteOptimizationHistoryCallback(tmp_path / "history.csv")
    callback.initialize(SimpleNamespace(problem=_Problem()))
    data = _data(None, [1.0], [[-1.0]])

    with pytest.raises(ValueError, match="iteration number"):
        callback.notify(data)
