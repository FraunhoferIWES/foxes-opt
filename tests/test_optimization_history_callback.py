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
