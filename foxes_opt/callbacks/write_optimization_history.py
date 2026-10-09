from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from iwopy.core import Optimizer, OptimizerCallback, OptimizerCallbackData, Problem


class WriteOptimizationHistoryCallback(OptimizerCallback):
    """Write objective and constraint-violation counts to a CSV file.

    Population states report the individual with the best finite configured
    objective. Iterations without cached values retain their row with empty
    objective and constraint fields. By default initialization replaces an
    existing file. Append mode preserves rows only when the existing CSV header
    matches this callback's schema.
    Enable ``write_initial`` to evaluate the starting variables and record
    their objective and constraint-violation count before optimizer iterations.
    """

    def __init__(
        self,
        file_path: str | Path,
        objective: int = 0,
        iteration_offset: int = 0,
        append: bool = False,
        write_initial: bool = False,
    ) -> None:
        """Initialize the optimization history callback.

        Parameters
        ----------
        file_path
            Path of the CSV history file.
        objective
            Objective component used to select and report an individual.
        iteration_offset
            Non-negative offset added to reported iteration numbers.
        append
            Whether to preserve and append to a non-empty existing history
            file. Missing and empty files receive a new header.
        write_initial
            Evaluate the initial variables during initialization and write their
            objective and constraint-violation count at ``iteration_offset``.
            Leave disabled when appending a restart whose starting row already
            exists.

        Raises
        ------
        TypeError
            If ``iteration_offset`` is not an integer or ``append`` or
            ``write_initial`` is not a boolean.
        ValueError
            If ``objective`` is not a non-negative integer or
            ``iteration_offset`` is negative.
        """
        super().__init__()
        if (
            isinstance(objective, bool)
            or not isinstance(objective, int)
            or objective < 0
        ):
            raise ValueError("objective must be a non-negative integer")
        if isinstance(iteration_offset, bool) or not isinstance(iteration_offset, int):
            raise TypeError("iteration_offset must be an integer")
        if iteration_offset < 0:
            raise ValueError("iteration_offset must be non-negative")
        if not isinstance(append, bool):
            raise TypeError("append must be a boolean")
        if not isinstance(write_initial, bool):
            raise TypeError("write_initial must be a boolean")
        self.file_path = Path(file_path)
        self.objective = objective
        self.iteration_offset = iteration_offset
        self.append = append
        self.write_initial = write_initial
        self._problem: Problem | None = None

    def initialize(self, optimizer: Optimizer) -> None:
        """Prepare the history file and optionally record the starting values.

        Parameters
        ----------
        optimizer
            Optimizer whose problem defines objective direction and constraint
            feasibility.

        Raises
        ------
        IndexError
            If the selected objective component does not exist.
        ValueError
            If append mode encounters a non-empty file with another CSV
            header, or initial output is enabled and required initial variables
            are undefined.
        """
        super().initialize(optimizer)
        self._problem = optimizer.problem
        if self.objective >= self._problem.n_objectives:
            raise IndexError(
                f"Objective index {self.objective} is outside "
                f"[0, {self._problem.n_objectives})"
            )
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        header = ["iteration", "objective", "n_violated_constraints"]
        if self.append and self.file_path.exists() and self.file_path.stat().st_size:
            with self.file_path.open(encoding="utf-8", newline="") as stream:
                existing_header = next(csv.reader(stream), None)
            if existing_header != header:
                raise ValueError(
                    f"Unexpected optimization history header in {self.file_path}"
                )
        else:
            with self.file_path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(header)
        if self.write_initial:
            self._write_initial_iteration()

    def _write_initial_iteration(self) -> None:
        """Evaluate initial optimization variables and append their history row."""
        if self._problem is None:
            raise RuntimeError("Optimization history callback has not been initialized")
        variables_int = np.empty(0, dtype=np.int32)
        if self._problem.n_vars_int:
            initial_int = self._problem.initial_values_int()
            if initial_int is None:
                raise ValueError("Initial integer variables are undefined")
            variables_int = np.asarray(initial_int)
        variables_float = np.empty(0, dtype=np.float64)
        if self._problem.n_vars_float:
            initial_float = self._problem.initial_values_float()
            if initial_float is None:
                raise ValueError("Initial float variables are undefined")
            variables_float = np.asarray(initial_float, dtype=np.float64)
        objectives, constraints = self._problem.evaluate_individual(
            variables_int, variables_float
        )
        self.notify(
            OptimizerCallbackData(
                event="iteration",
                iteration=0,
                vars_int=variables_int,
                vars_float=variables_float,
                objs=objectives,
                cons=constraints,
            )
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
        """Append the current objective and violation count to the CSV file.

        Parameters
        ----------
        data
            Normalized optimizer callback snapshot for one iteration.

        Raises
        ------
        ValueError
            If the snapshot has no iteration number or lacks the configured
            objective component.
        RuntimeError
            If the callback has not been initialized.
        """
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
                [
                    self.iteration_offset + data.iteration,
                    objective_value,
                    n_violated_constraints,
                ]
            )
