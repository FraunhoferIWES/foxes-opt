from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import foxes
import foxes.variables as FV
import numpy as np
from iwopy.core import Optimizer, OptimizerCallback, OptimizerCallbackData

from foxes_opt.core import FarmConstraint, FarmOptProblem


class WriteLayoutCallback(OptimizerCallback):
    """Write intermediate optimization layouts to CSV and image files.

    By default, the initial layout is written as step zero before optimizer
    iterations begin. This can be disabled with ``write_initial=False``.
    For population snapshots, the member with the best finite first objective
    is written. Iteration-based optimizers use their iteration number in file
    names; evaluation-based optimizers use their cumulative evaluation count.

    """

    def __init__(
        self,
        out_dir: str | Path,
        base_name: str,
        n_step: int = 1,
        figsize: tuple[float, float] | None = None,
        verbosity: int = 0,
        write_csv: bool = True,
        write_image: bool = True,
        image_format: str = "jpg",
        valid_color: str = "tab:blue",
        invalid_color: str = "red",
        write_initial: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        out_dir
            The output directory for CSV layouts. Images are written to a
            child directory named after their file format.
        base_name
            The base name for the layout files.
        n_step
            The number of iterations or evaluations between writing layouts.
        figsize
            The figure size for layout plots.
        verbosity
            The verbosity level, 0 = silent.
        write_csv
            Whether to write CSV layout files.
        write_image
            Whether to write layout plot files.
        image_format
            The image file extension used for layout plots.
        valid_color
            The Matplotlib color for valid turbines.
        invalid_color
            The Matplotlib color for turbines associated with violated
            constraints.
        write_initial
            Whether to write the initial layout as step zero.
        """
        super().__init__()
        if n_step < 1:
            raise ValueError("n_step must be at least 1")
        image_format = image_format.removeprefix(".").lower()
        if not image_format or not image_format.isalnum():
            raise ValueError("image_format must be a file extension")
        self.out_dir = Path(out_dir)
        self.base_name = base_name
        self.n_step = n_step
        self.figsize = figsize
        self.verbosity = verbosity
        self.write_csv = write_csv
        self.write_image = write_image
        self.image_format = image_format
        self.valid_color = valid_color
        self.invalid_color = invalid_color
        self.write_initial = write_initial
        self._problem: FarmOptProblem | None = None
        self._farm: foxes.WindFarm | None = None

    def initialize(self, optimizer: Optimizer) -> None:
        """Prepare a separate farm for writing layouts.

        Parameters
        ----------
        optimizer
            The optimizer starting the run.
        """
        super().initialize(optimizer)
        problem = optimizer.problem
        while hasattr(problem, "base_problem"):
            problem = problem.base_problem
        if not isinstance(problem, FarmOptProblem):
            raise TypeError(
                f"Expected problem of type {FarmOptProblem.__name__}, "
                f"got {type(problem).__name__}"
            )
        self._problem = problem
        self._farm = deepcopy(self._problem.farm)
        diameters = self._problem.farm.get_rotor_diameters(self._problem.algo)
        for turbine, diameter in zip(self._farm.turbines, diameters, strict=True):
            turbine.D = float(diameter)
        if self.write_csv or self.write_image:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            if self.write_initial:
                self._write_layout(0, None, set())

    def _best_index(self, data: OptimizerCallbackData) -> int:
        if len(data.vars_float) == 1:
            return 0
        if self._problem is None:
            raise RuntimeError("Layout callback has not been initialized")
        if data.objs is None or not data.objs.shape[1]:
            raise ValueError("Population layout snapshots require objective values")
        objective = data.objs[:, 0]
        finite = np.flatnonzero(np.isfinite(objective))
        if not len(finite):
            raise ValueError("Layout snapshots require a finite objective value")
        select = np.argmax if self._problem.maximize_objs[0] else np.argmin
        return int(finite[select(objective[finite])])

    def _set_layout(self, data: OptimizerCallbackData) -> int:
        if self._problem is None or self._farm is None:
            raise RuntimeError("Layout callback has not been initialized")
        selected = self._best_index(data)
        coordinates = data.vars_float[selected]
        expected_size = 2 * self._problem.n_sel_turbines
        if coordinates.size != expected_size:
            raise ValueError(
                f"Expected {expected_size} layout variables, got {coordinates.size}"
            )
        coordinates = coordinates.reshape(self._problem.n_sel_turbines, 2)
        for turbine_i, position in zip(self._problem.sel_turbines, coordinates):
            self._farm.turbines[turbine_i].xy = position.copy()
        return selected

    def _invalid_turbines(self, data: OptimizerCallbackData, selected: int) -> set[int]:
        if self._problem is None:
            raise RuntimeError("Layout callback has not been initialized")
        if data.cons is None or not data.cons.shape[1]:
            return set()

        valid = self._problem.check_constraints_individual(data.cons[selected])
        invalid: set[int] = set()
        component = 0
        for constraint in self._problem.cons.functions:
            next_component = component + constraint.n_components()
            violated = np.flatnonzero(~valid[component:next_component])
            if isinstance(constraint, FarmConstraint):
                invalid.update(constraint.component_turbines(violated))
            else:
                dependencies = np.asarray(constraint.vardeps_float(), dtype=bool)
                variable_names = np.asarray(constraint.var_names_float)
                dependent = np.any(dependencies[violated], axis=0)
                for variable_name in variable_names[dependent]:
                    try:
                        variable, turbine_i = self._problem.parse_tvar(variable_name)
                    except (IndexError, ValueError):
                        continue
                    if variable in (FV.X, FV.Y):
                        invalid.add(turbine_i)
            component = next_component
        return invalid

    def _write_layout(
        self, step: int, objective: float | None, invalid_turbines: set[int]
    ) -> None:
        if self._problem is None or self._farm is None:
            raise RuntimeError("Layout callback has not been initialized")
        basename = self.out_dir / f"{self.base_name}_{step:05d}"
        output = foxes.output.FarmLayoutOutput(self._farm)
        if self.write_csv:
            csv_path = basename.with_suffix(".csv")
            if self.verbosity > 0:
                print(f"Writing layout to {csv_path}")
            output.write_csv(
                str(csv_path),
                verbosity=0,
                type_col="turbine_type",
                algo=self._problem.algo,
            )
        if self.write_image:
            image_path = (
                self.out_dir
                / self.image_format
                / basename.with_suffix(f".{self.image_format}").name
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            objective_name = self._problem.objs.component_names[0]
            objective_value = (
                "unavailable"
                if objective is None or not np.isfinite(objective)
                else f"{objective:.6g}"
            )
            if self.verbosity > 0:
                print(f"Writing layout to {image_path}")
            colors = np.full(self._farm.n_turbines, self.valid_color, dtype=object)
            colors[list(invalid_turbines)] = self.invalid_color
            output.write_plot(
                str(image_path),
                figsize=self.figsize,
                annotate=0,
                true_turbine_radii=True,
                title=f"{objective_name}: {objective_value}",
                c=colors,
                edgecolors=colors,
                linewidths=0.4,
                zorder=5,
                legend_labels={
                    self.valid_color: "Valid turbine",
                    self.invalid_color: "Constraint violation",
                },
            )

    def notify(self, data: OptimizerCallbackData) -> None:
        """Write a layout for an intermediate optimizer state.

        Parameters
        ----------
        data
            The current normalized optimizer state.
        """
        if not len(data.vars_float) or not (self.write_csv or self.write_image):
            return
        step = data.iteration
        if step is None:
            step = data.n_evaluations
        if step is None:
            raise ValueError(
                "Layout snapshots require an iteration or evaluation count"
            )
        if step % self.n_step:
            return
        selected = self._set_layout(data)
        objective = None
        if data.objs is not None and data.objs.shape[1]:
            objective = float(data.objs[selected, 0])
        invalid_turbines = self._invalid_turbines(data, selected)
        self._write_layout(step, objective, invalid_turbines)
