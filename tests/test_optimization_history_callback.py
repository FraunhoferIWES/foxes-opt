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
