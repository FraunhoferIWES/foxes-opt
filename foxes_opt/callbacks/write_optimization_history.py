from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from iwopy.core import Optimizer, OptimizerCallback, OptimizerCallbackData, Problem


class WriteOptimizationHistoryCallback(OptimizerCallback):
    """Write objective and constraint-violation counts to a CSV file.

    Population states report the individual with the best finite configured
    objective. Iterations without cached values retain their row with empty
    objective and constraint fields.
    """

    def __init__(self, file_path: str | Path, objective: int = 0) -> None:
        """
        Parameters
        ----------
        file_path
            Path of the CSV history file.
        objective
            Objective component used to select and report an individual.
        """
        super().__init__()
        if (
            isinstance(objective, bool)
            or not isinstance(objective, int)
            or objective < 0
        ):
            raise ValueError("objective must be a non-negative integer")
        self.file_path = Path(file_path)
        self.objective = objective
        self._problem: Problem | None = None

    def initialize(self, optimizer: Optimizer) -> None:
        """Initialize the callback and reset the history file."""
        super().initialize(optimizer)
        self._problem = optimizer.problem
        if self.objective >= self._problem.n_objectives:
            raise IndexError(
                f"Objective index {self.objective} is outside "
                f"[0, {self._problem.n_objectives})"
            )
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(
                ["iteration", "objective", "n_violated_constraints"]
            )

    def _best_index(self, data: OptimizerCallbackData) -> int | None:
        if data.objs is None:
            return None
        if self._problem is None:
            raise RuntimeError("Optimization history callback has not been initialized")
        if self.objective >= data.objs.shape[1]:
            raise ValueError(
                f"Expected objective component {self.objective}, "
                f"got {data.objs.shape[1]} components"
            )
        objective_values = data.objs[:, self.objective]
        finite = np.flatnonzero(np.isfinite(objective_values))
        if not len(finite):
            return None
        select = np.argmax if self._problem.maximize_objs[self.objective] else np.argmin
        return int(finite[select(objective_values[finite])])

    def notify(self, data: OptimizerCallbackData) -> None:
        """Append the current objective and violation count to the CSV file."""
        if data.iteration is None:
            raise ValueError("Optimization history output requires an iteration number")
        if self._problem is None:
            raise RuntimeError("Optimization history callback has not been initialized")

        selected = self._best_index(data)
        objective_value: float | None = None
        n_violated_constraints: int | None = None
        if selected is not None:
            objective_value = float(data.objs[selected, self.objective])
            if data.cons is not None:
                valid = self._problem.check_constraints_individual(data.cons[selected])
                n_violated_constraints = int(np.count_nonzero(~valid))

        with self.file_path.open("a", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerow(
                [data.iteration, objective_value, n_violated_constraints]
            )
