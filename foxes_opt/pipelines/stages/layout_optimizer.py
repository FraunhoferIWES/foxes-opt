from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from iwopy import Pipeline
from iwopy.core import Optimizer, PipelineStage, Problem
from iwopy.utils import new_instance
from iwopy.wrappers import ProblemWrapper

from foxes_opt.constraints import FarmBoundaryConstraint, MinDistConstraint
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
        min_dist: float = 2.5,
        min_dist_unit: str = "D",
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
        self.min_dist = min_dist
        self.min_dist_unit = min_dist_unit
        self.flow_states = flow_states

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
        if not self.problem_type:
            raise ValueError(f"{self.name}: Missing problem_type")
        if self.min_dist <= 0:
            raise ValueError(f"{self.name}: min_dist must be positive")
        if self.min_dist_unit not in ("m", "D"):
            raise ValueError(f"{self.name}: min_dist_unit must be either 'm' or 'D'")
        if {"problem"}.intersection(self.optimizer_pars):
            raise ValueError(
                f"{self.name}: optimizer_pars contains reserved parameter 'problem'"
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

    def _add_functions(self, problem: FarmOptProblem) -> None:
        """Add the configured objectives and constraints to a problem."""
        if self.objectives is None:
            problem.add_objective(
                MaxFarmREWS(
                    problem,
                    sel_turbines=list(range(problem.farm.n_turbines)),
                )
            )
        for pars in self.objectives or []:
            problem.add_objective(FarmObjective.new(problem=problem, **pars))
        if self.constraints is None:
            problem.add_constraint(FarmBoundaryConstraint(problem))
            problem.add_constraint(
                MinDistConstraint(
                    problem,
                    min_dist=self.min_dist,
                    min_dist_unit=self.min_dist_unit,
                )
            )
        for pars in self.constraints or []:
            problem.add_constraint(FarmConstraint.new(problem=problem, **pars))

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

    def _create_optimizer(self, problem: Problem) -> Optimizer:
        """Create the configured iwopy optimizer."""
        return Optimizer.new(
            problem=problem,
            optimizer_type=self.optimizer_type,
            **self.optimizer_pars,
        )

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
        tuple[Any, np.ndarray]
            The iwopy optimizer results and the candidate layout coordinates.
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
            name=f"{self.name}_problem",
            algo=algo,
            sel_turbines=sel_turbines,
            **self.problem_pars,
        )
        self._add_functions(problem)
        optimizer_problem = self._prepare_problem(problem)
        optimizer_problem.initialize(verbosity=max(verbosity - 2, 0))
        optimizer = self._create_optimizer(optimizer_problem)
        optimizer.initialize(verbosity=max(verbosity - 2, 0))
        try:
            results = optimizer.solve(verbosity=max(verbosity - 1, 0))
            optimizer.finalize(results, verbosity=max(verbosity - 1, 0))
            candidate = layout_xy.copy()
            selected = problem.sel_turbines
            candidate[selected] = results.vars_float.reshape(-1, 2)
            return results, candidate
        finally:
            if algo.initialized and not algo.running:
                algo.finalize()

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
        tuple[bool, np.ndarray]
            ``True`` and the optimized layout when the optimizer succeeds;
            otherwise the original layout together with ``False``.
        """
        del prev_stage, layout_plot_pars, kwargs
        layout_xy = self._pipeline.read_layout(prev_results).copy()
        results, candidate = self._run_layout_optimizer(layout_xy, verbosity=verbosity)
        success = bool(results.success) and np.all(np.isfinite(candidate))
        return success, candidate if success else layout_xy
