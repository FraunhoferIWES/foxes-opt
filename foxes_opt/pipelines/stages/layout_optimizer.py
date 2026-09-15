from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from iwopy.core import Optimizer, Pipeline, PipelineStage, Problem
from iwopy.utils import new_instance
from iwopy.wrappers import ProblemWrapper

from foxes_opt.constraints import FarmBoundaryConstraint
from foxes_opt.core import FarmConstraint, FarmObjective, FarmOptProblem
from foxes_opt.objectives import MaxFarmREWS

if TYPE_CHECKING:
    from foxes.core import Algorithm, States


class LayoutOptimizerStage(PipelineStage):
    """
    Optimize the layout received from the previous pipeline stage.

    Objectives, constraints, and the optimizer are configured through iwopy's
    runtime factories. Subclasses can specialize problem preparation, for
    example by adding a gradient wrapper or selecting turbines and states.
    """

    def __init__(
        self,
        optimizer_type: str,
        optimizer_pars: dict[str, Any] | None = None,
        problem_type: str = "FarmLayoutOptProblem",
        objectives: list[dict[str, Any]] | None = None,
        constraints: list[dict[str, Any]] | None = None,
        problem_pars: dict[str, Any] | None = None,
        problem_wrapper_type: str | None = None,
        problem_wrapper_pars: dict[str, Any] | None = None,
        min_dist: float | list[float] | None = 2.5,
        min_dist_unit: str = "D",
        min_dist_constraint_type: str | None = "MinDistConstraint",
        boundary_repair: bool = False,
        boundary_repair_optimizer_type: str | None = None,
        boundary_repair_optimizer_pars: dict[str, Any] | None = None,
        flow_states: States | None = None,
        name: str = "layout_optimizer",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        optimizer_type
            The iwopy optimizer type passed to ``Optimizer.new``.
        optimizer_pars
            Additional parameters for the selected optimizer.
        problem_type
            The ``FarmOptProblem`` subclass name passed to
            ``FarmOptProblem.new``.
        objectives
            Objective factory parameters. Every entry requires
            ``objective_type``.
        constraints
            Constraint factory parameters. Every entry requires
            ``constraint_type``.
        problem_pars
            Additional parameters for the selected optimization problem.
        problem_wrapper_type
            Optional iwopy problem wrapper type, for example ``"LocalFD"``.
        problem_wrapper_pars
            Parameters for the optional problem wrapper.
        min_dist
            Minimum distance used by the default turbine-distance constraint.
        min_dist_unit
            Unit of the default minimum distance, either ``"m"`` or ``"D"``.
        min_dist_constraint_type
            Constraint type used for the default minimum-distance constraint,
            or ``None``/``"None"`` to skip it.
        boundary_repair
            If ``True``, first optimize turbines that violate the farm boundary,
            then exclude them from the main optimization.
        boundary_repair_optimizer_type
            Optimizer type for the boundary repair run. If ``None``, use
            ``optimizer_type``.
        boundary_repair_optimizer_pars
            Optimizer parameters for the boundary repair run. These parameters
            override ``optimizer_pars`` for the repair run only.
        flow_states
            States used for optimization, or ``None`` for pipeline states.
        name
            Stage name.
        kwargs
            Additional parameters for ``PipelineStage``.
        """
        super().__init__(name=name, **kwargs)
        self.optimizer_type = optimizer_type
        self.optimizer_pars = {} if optimizer_pars is None else optimizer_pars.copy()
        self.problem_type = problem_type
        self.objectives = (
            None if objectives is None else [pars.copy() for pars in objectives]
        )
        self.constraints = (
            None if constraints is None else [pars.copy() for pars in constraints]
        )
        self.problem_pars = {} if problem_pars is None else problem_pars.copy()
        self.problem_wrapper_type = problem_wrapper_type
        self.problem_wrapper_pars = (
            {} if problem_wrapper_pars is None else problem_wrapper_pars.copy()
        )
        self.__min_dist: float | list[float] | None = min_dist
        self.min_dist_unit = min_dist_unit
        self.min_dist_constraint_type = min_dist_constraint_type
        self.boundary_repair = boundary_repair
        self.boundary_repair_optimizer_type = boundary_repair_optimizer_type
        self.boundary_repair_optimizer_pars = (
            None
            if boundary_repair_optimizer_pars is None
            else boundary_repair_optimizer_pars.copy()
        )
        self.flow_states = flow_states

    @property
    def min_dist(self) -> float | list[float] | None:
        return self.__min_dist

    def initialize(self, pipeline: Pipeline, verbosity: int = 0) -> None:
        """
        Initialize and validate the stage against its layout pipeline.

        Parameters
        ----------
        pipeline
            The pipeline hosting this optimization stage.
        verbosity
            Verbosity level used during initialization.
        """
        super().initialize(pipeline, verbosity=verbosity)
        self._pipeline = pipeline
        self._flow_states = (
            pipeline.states if self.flow_states is None else self.flow_states
        )
        if self._flow_states is None:
            raise ValueError(f"{self.name}: Missing flow states")
        if pipeline.farm_boundary is None:
            raise ValueError(f"{self.name}: A farm boundary is required")
        if not self.optimizer_type:
            raise ValueError(f"{self.name}: Missing optimizer_type")
        if self.boundary_repair_optimizer_type is not None and not (
            self.boundary_repair_optimizer_type
        ):
            raise ValueError(
                f"{self.name}: boundary_repair_optimizer_type must not be empty"
            )
        if not self.problem_type:
            raise ValueError(f"{self.name}: Missing problem_type")
        if self.min_dist_unit not in ("m", "D"):
            raise ValueError(f"{self.name}: min_dist_unit must be either 'm' or 'D'")
        if {"problem"}.intersection(self.optimizer_pars):
            raise ValueError(
                f"{self.name}: optimizer_pars contains reserved parameter 'problem'"
            )
        if self.boundary_repair_optimizer_pars is not None and {"problem"}.intersection(
            self.boundary_repair_optimizer_pars
        ):
            raise ValueError(
                f"{self.name}: boundary_repair_optimizer_pars contains reserved parameter 'problem'"
            )
        if {"name", "algo", "sel_turbines"}.intersection(self.problem_pars):
            raise ValueError(
                f"{self.name}: problem_pars contains reserved problem parameters"
            )
        if self.problem_wrapper_type is None and self.problem_wrapper_pars:
            raise ValueError(
                f"{self.name}: problem_wrapper_pars requires problem_wrapper_type"
            )
        if {"base_problem"}.intersection(self.problem_wrapper_pars):
            raise ValueError(
                f"{self.name}: problem_wrapper_pars contains reserved parameter 'base_problem'"
            )
        for pars in self.objectives or []:
            if "objective_type" not in pars:
                raise ValueError(
                    f"{self.name}: Every objective requires objective_type"
                )
        for pars in self.constraints or []:
            if "constraint_type" not in pars:
                raise ValueError(
                    f"{self.name}: Every constraint requires constraint_type"
                )

    def _add_objectives(self, problem: FarmOptProblem) -> None:
        """
        Add the configured objectives to a problem.

        Parameters
        ----------
        problem
            The optimization problem
        """
        if self.objectives is None:
            problem.add_objective(
                MaxFarmREWS(
                    problem,
                    sel_turbines=list(range(problem.farm.n_turbines)),
                )
            )
        for pars in self.objectives or []:
            problem.add_objective(FarmObjective.new(problem=problem, **pars))

    def _add_main_constraints(self, problem: FarmOptProblem) -> None:
        """
        Add the configured main optimization constraints to a problem.

        Parameters
        ----------
        problem
            The optimization problem
        """
        if self.constraints is None:
            problem.add_constraint(FarmBoundaryConstraint(problem))
        if self.min_dist is not None and self.min_dist_constraint_type not in (
            None,
            "None",
        ):
            problem.add_constraint(
                FarmConstraint.new(
                    self.min_dist_constraint_type,
                    problem=problem,
                    min_dist=self.min_dist,
                    min_dist_unit=self.min_dist_unit,
                )
            )
        for pars in self.constraints or []:
            problem.add_constraint(FarmConstraint.new(problem=problem, **pars))

    def _add_functions(self, problem: FarmOptProblem) -> None:
        """
        Add the configured objectives and constraints to a problem.

        Parameters
        ----------
        problem
            The optimization problem
        """
        self._add_objectives(problem)
        self._add_main_constraints(problem)

    def _add_boundary_repair_functions(self, problem: FarmOptProblem) -> None:
        """
        Add objectives and repair-only constraints to a problem.

        Parameters
        ----------
        problem
            The optimization problem
        """
        self._add_objectives(problem)
        problem.add_constraint(FarmBoundaryConstraint(problem))
        if self.min_dist is not None and self.min_dist_constraint_type not in (
            None,
            "None",
        ):
            problem.add_constraint(
                FarmConstraint.new(
                    self.min_dist_constraint_type,
                    problem=problem,
                    min_dist=self.min_dist,
                    min_dist_unit=self.min_dist_unit,
                    check_only_selected=True,
                )
            )

    def _active_turbines(
        self, layout_xy: np.ndarray, sel_turbines: list[int] | None
    ) -> list[int]:
        """
        Return the active turbine list for an optimization call.

        Parameters
        ----------
        layout_xy
            The turbine coordinates
        sel_turbines
            The selected turbines, or ``None`` for all turbines

        Returns
        -------
        turbines
            The active turbine indices
        """
        return (
            list(range(len(layout_xy))) if sel_turbines is None else sel_turbines.copy()
        )

    def _boundary_repair_thresholds(
        self,
        layout_xy: np.ndarray,
        states: States | None,
        sel_turbines: list[int],
        verbosity: int,
    ) -> np.ndarray:
        """
        Calculate the boundary repair thresholds for selected turbines.

        Parameters
        ----------
        layout_xy
            The turbine coordinates
        states
            Optional states used for the optimization
        sel_turbines
            The selected turbines
        verbosity
            Verbosity level used for temporary algorithm setup

        Returns
        -------
        thresholds
            The repair thresholds in metres
        """
        min_dist: float | list[float]
        if self.min_dist is None:
            min_dist = 1.0
            min_dist_unit = "D"
        else:
            min_dist = self.min_dist
            min_dist_unit = self.min_dist_unit

        threshold = np.asarray(min_dist, dtype=float)
        if threshold.size == 1:
            threshold = np.full(len(sel_turbines), threshold.item())
        elif threshold.size != len(sel_turbines):
            raise ValueError(
                f"{self.name}: Boundary repair threshold has length {threshold.size}, expected 1 or {len(sel_turbines)}"
            )

        if min_dist_unit == "m":
            return threshold

        algo: Algorithm = self._pipeline.get_algo(
            layout_xy=layout_xy,
            states=self._flow_states if states is None else states,
            initialize=True,
            force=False,
            verbosity=max(verbosity - 2, 0),
        )
        try:
            diameters = algo.farm.get_rotor_diameters(algo)
            return threshold * np.asarray(diameters, dtype=float)[sel_turbines]
        finally:
            if algo.initialized and not algo.running:
                algo.finalize()

    def _boundary_repair_turbines(
        self,
        layout_xy: np.ndarray,
        states: States | None,
        sel_turbines: list[int],
        verbosity: int,
    ) -> list[int]:
        """
        Return active turbines requiring boundary repair.

        Parameters
        ----------
        layout_xy
            The turbine coordinates
        states
            Optional states used for the optimization
        sel_turbines
            The selected turbines
        verbosity
            Verbosity level used for temporary algorithm setup

        Returns
        -------
        turbines
            The turbine indices that violate or are close to the farm boundary
        """
        xy = layout_xy[sel_turbines]
        boundary = self._pipeline.farm_boundary
        dists = boundary.points_distance(xy)
        inside = boundary.points_inside(xy)
        signed_dists = dists.copy()
        signed_dists[inside] *= -1
        thresholds = self._boundary_repair_thresholds(
            layout_xy,
            states,
            sel_turbines,
            verbosity,
        )
        repair = (signed_dists > 0) | (dists < thresholds)
        return np.asarray(sel_turbines, dtype=int)[repair].tolist()

    def _prepare_problem(self, problem: FarmOptProblem) -> Problem:
        """Return the initialized problem supplied to the optimizer."""
        if self.problem_wrapper_type is None:
            return problem
        return new_instance(
            ProblemWrapper,
            self.problem_wrapper_type,
            base_problem=problem,
            **self.problem_wrapper_pars,
        )

    def _create_optimizer(
        self,
        problem: Problem,
        optimizer_type: str | None = None,
        optimizer_pars: dict[str, Any] | None = None,
    ) -> Optimizer:
        """
        Create the configured iwopy optimizer.

        Parameters
        ----------
        problem
            The problem supplied to the optimizer
        optimizer_type
            Optional optimizer type override
        optimizer_pars
            Optional optimizer parameter overrides

        Returns
        -------
        optimizer
            The configured optimizer
        """
        pars = self.optimizer_pars.copy()
        if optimizer_pars is not None:
            pars.update(optimizer_pars)
        return Optimizer.new(
            problem=problem,
            optimizer_type=self.optimizer_type
            if optimizer_type is None
            else optimizer_type,
            **pars,
        )

    def _solve_layout_problem(
        self,
        layout_xy: np.ndarray,
        states: States | None,
        sel_turbines: list[int] | None,
        add_functions: Callable[[FarmOptProblem], None],
        problem_name: str,
        optimizer_type: str | None,
        optimizer_pars: dict[str, Any] | None,
        verbosity: int,
    ) -> tuple[Any, np.ndarray]:
        """
        Create, solve, and finalize one layout optimization problem.

        Parameters
        ----------
        layout_xy
            Current turbine coordinates
        states
            Optional states used for the optimization
        sel_turbines
            Turbine indices to optimize, or ``None`` for all turbines
        add_functions
            Function that adds objectives and constraints to the problem
        problem_name
            The optimization problem name
        optimizer_type
            Optional optimizer type override
        optimizer_pars
            Optional optimizer parameter overrides
        verbosity
            Verbosity level passed to the algorithm and optimizer setup

        Returns
        -------
        results, candidate
            The iwopy optimizer results and the candidate layout coordinates
        """
        algo: Algorithm = self._pipeline.get_algo(
            layout_xy=layout_xy,
            states=self._flow_states if states is None else states,
            initialize=False,
            force=False,
            verbosity=0,
        )
        problem = FarmOptProblem.new(
            problem_type=self.problem_type,
            name=problem_name,
            algo=algo,
            sel_turbines=sel_turbines,
            **self.problem_pars,
        )
        add_functions(problem)
        optimizer_problem = self._prepare_problem(problem)
        optimizer_problem.initialize(verbosity=max(verbosity - 1, 0))
        optimizer = self._create_optimizer(
            optimizer_problem,
            optimizer_type,
            optimizer_pars,
        )
        optimizer.initialize(verbosity=max(verbosity - 2, 0))
        if verbosity > 1:
            optimizer.print_info()
        try:
            results = optimizer.solve(verbosity=max(verbosity - 1, 0))
            optimizer.finalize(results, verbosity=max(verbosity - 1, 0))
            candidate = layout_xy.copy()
            selected = problem.sel_turbines
            if not results.success:
                return results, candidate
            if getattr(problem, "n_vars_int", 0):
                if results.vars_int is None:
                    return results, candidate
                problem.update_problem_individual(results.vars_int, results.vars_float)
                candidate[selected] = np.array(
                    [algo.farm.turbines[ti].xy for ti in selected]
                )
            else:
                if results.vars_float is None:
                    return results, candidate
                candidate[selected] = results.vars_float.reshape(-1, 2)
            return results, candidate
        finally:
            if algo.initialized and not algo.running:
                algo.finalize()

    def _run_layout_optimizer(
        self,
        layout_xy: np.ndarray,
        states: States | None = None,
        sel_turbines: list[int] | None = None,
        verbosity: int = 1,
    ) -> tuple[Any, np.ndarray]:
        """
        Optimize selected layout coordinates and return solver results.

        Parameters
        ----------
        layout_xy
            Current turbine coordinates.
        states
            Optional states used for the optimization. If ``None``, the stage
            flow states are used.
        sel_turbines
            Turbine indices to optimize. If ``None``, all turbines are
            considered.
        verbosity
            Verbosity level passed to the algorithm and optimizer setup.

        Returns
        -------
        results, candidate
            The iwopy optimizer results and the candidate layout coordinates.
        """
        active_turbines = self._active_turbines(layout_xy, sel_turbines)
        if self.boundary_repair:
            repair_turbines = self._boundary_repair_turbines(
                layout_xy,
                states,
                active_turbines,
                verbosity,
            )
            if repair_turbines:
                repair_results, repaired_layout = self._solve_layout_problem(
                    layout_xy,
                    states,
                    repair_turbines,
                    self._add_boundary_repair_functions,
                    f"{self.name}_boundary_repair_problem",
                    self.boundary_repair_optimizer_type,
                    self.boundary_repair_optimizer_pars,
                    verbosity,
                )
                if not repair_results.success or not np.all(
                    np.isfinite(repaired_layout)
                ):
                    return repair_results, layout_xy.copy()
                layout_xy = repaired_layout
                remaining_turbines = [
                    ti for ti in active_turbines if ti not in repair_turbines
                ]
                if not remaining_turbines:
                    return repair_results, layout_xy
                sel_turbines = remaining_turbines

        return self._solve_layout_problem(
            layout_xy,
            states,
            sel_turbines,
            self._add_functions,
            f"{self.name}_problem",
            None,
            None,
            verbosity,
        )

    def run(
        self,
        prev_stage: PipelineStage | None = None,
        prev_results: Any = None,
        layout_plot_pars: dict[str, Any] | None = None,
        verbosity: int = 1,
        **kwargs: Any,
    ) -> tuple[bool, np.ndarray]:
        """
        Run the configured optimizer from the previous layout.

        Parameters
        ----------
        prev_stage
            Previous pipeline stage, unused by this implementation.
        prev_results
            Results from the previous stage, used to read the current layout.
        layout_plot_pars
            Optional plotting parameters, unused here.
        verbosity
            Verbosity level for the optimization run.
        kwargs
            Additional keyword arguments, unused by this stage.

        Returns
        -------
        success, layout_xy
            ``True`` and the optimized layout when the optimizer succeeds;
            otherwise the original layout together with ``False``.
        """
        del prev_stage, layout_plot_pars, kwargs
        layout_xy = self._pipeline.read_layout(prev_results).copy()
        results, candidate = self._run_layout_optimizer(layout_xy, verbosity=verbosity)
        success = bool(results.success) and np.all(np.isfinite(candidate))
        return success, candidate if success else layout_xy
