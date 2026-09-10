from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from foxes.core import SubsetStates, run_with_engine
from iwopy import Pipeline
from iwopy.core import PipelineStage

from .layout_optimizer import LayoutOptimizerStage

if TYPE_CHECKING:
    from foxes.core import States


class RandomSubsetStage(LayoutOptimizerStage):
    """
    Repeated layout optimization on random turbine and state subsets.

    Each step independently samples turbines and states without replacement.
    The configured iwopy optimizer receives a layout problem for the sampled
    turbines and states. The next step starts from the prior layout only when
    the optimizer returns a valid, strictly improved result.
    """

    def __init__(
        self,
        n_turbines: int,
        n_states: int,
        n_steps: int,
        optimizer_type: str,
        optimizer_pars: dict[str, Any] | None = None,
        problem_wrapper_type: str | None = None,
        problem_wrapper_pars: dict[str, Any] | None = None,
        min_dist: float = 2.5,
        min_dist_unit: str = "D",
        seed: int | None = None,
        flow_states: States | None = None,
        write_step_results: bool = False,
        name: str = "random_subset",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        n_turbines
            Number of movable turbines in each optimization step.
        n_states
            Number of active states in each optimization step.
        n_steps
            Number of successive random-subset optimizations.
        optimizer_type
            The iwopy optimizer type.
        optimizer_pars
            Parameters for the selected optimizer.
        problem_wrapper_type
            Optional iwopy problem wrapper type, such as ``"LocalFD"``.
        problem_wrapper_pars
            Parameters for the optional problem wrapper.
        min_dist
            Minimum turbine distance.
        min_dist_unit
            Unit of the minimum distance, either ``"m"`` or ``"D"``.
        seed
            Random seed. The generator is reset for every stage run.
        flow_states
            States from which active subsets are sampled, or ``None`` for the
            pipeline states.
        write_step_results
            Write full-state pipeline outputs after every accepted step.
        name
            Stage name.
        kwargs
            Additional parameters for ``LayoutOptimizerStage``.
        """
        super().__init__(
            optimizer_type=optimizer_type,
            optimizer_pars=optimizer_pars,
            problem_wrapper_type=problem_wrapper_type,
            problem_wrapper_pars=problem_wrapper_pars,
            min_dist=min_dist,
            min_dist_unit=min_dist_unit,
            flow_states=flow_states,
            name=name,
            **kwargs,
        )
        self.n_turbines = n_turbines
        self.n_states = n_states
        self.n_steps = n_steps
        self.seed = seed
        self.write_step_results = write_step_results

    def initialize(self, pipeline: Pipeline, verbosity: int = 0) -> None:
        """Initialize and validate random subset settings."""
        super().initialize(pipeline, verbosity=verbosity)
        if not 0 < self.n_turbines <= pipeline.n_turbines:
            raise ValueError(
                f"{self.name}: n_turbines must be in [1, {pipeline.n_turbines}], got {self.n_turbines}"
            )
        self._n_flow_states = self._flow_states.size()
        if self.n_states <= 0:
            raise ValueError(f"{self.name}: n_states must be positive")
        if self._n_flow_states and self.n_states > self._n_flow_states:
            raise ValueError(
                f"{self.name}: n_states must be in [1, {self._n_flow_states}], got {self.n_states}"
            )
        if self.n_steps <= 0:
            raise ValueError(f"{self.name}: n_steps must be positive")

    def _sample_subsets(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw one turbine subset and one state subset."""
        turbine_indices = rng.choice(
            self._pipeline.n_turbines, size=self.n_turbines, replace=False
        )
        state_indices = rng.choice(
            self._n_flow_states, size=self.n_states, replace=False
        )
        return turbine_indices, state_indices

    def _ensure_state_count(self, layout_xy: np.ndarray) -> None:
        """Initialize lazy states once to determine their full size."""
        if self._n_flow_states:
            return
        algo = self._pipeline.get_algo(
            layout_xy=layout_xy,
            states=self._flow_states,
            initialize=True,
            force=False,
            verbosity=0,
        )
        try:
            self._n_flow_states = algo.n_states
        finally:
            algo.finalize()
        if self.n_states > self._n_flow_states:
            raise ValueError(
                f"{self.name}: n_states must be in [1, {self._n_flow_states}], got {self.n_states}"
            )

    def _optimize_step(
        self,
        layout_xy: np.ndarray,
        turbine_indices: np.ndarray,
        state_indices: np.ndarray,
        verbosity: int,
    ) -> tuple[bool, np.ndarray]:
        """Run the selected optimizer for one sampled layout problem."""
        results, candidate = self._run_layout_optimizer(
            layout_xy,
            states=SubsetStates(self._flow_states, state_indices),
            sel_turbines=turbine_indices.tolist(),
            verbosity=verbosity,
        )
        accepted = bool(results.success) and np.all(np.isfinite(results.vars_float))
        return accepted, candidate if accepted else layout_xy

    def _write_step(
        self,
        layout_xy: np.ndarray,
        step: int,
        layout_plot_pars: dict[str, Any] | None,
        verbosity: int,
    ) -> None:
        """Write full-state outputs for an accepted optimization step."""
        pipeline = self._pipeline
        algo, farm_results = pipeline.run_foxes(
            layout_xy, force=True, verbosity=max(verbosity - 1, 0)
        )
        table_index = pipeline.add_to_table(
            f"{self.name}_{step}", True, algo, farm_results
        )
        pipeline.write_layout_csv(algo, farm_results, table_index=table_index)
        plot_pars = {
            "figsize": (12, 8),
            "show_flow": False,
            "resolution": 50.0,
            "annotate": 0,
        }
        if layout_plot_pars is not None:
            plot_pars.update(layout_plot_pars)
        pipeline.write_layout_plot(
            algo, farm_results, table_index=table_index, **plot_pars
        )

    def run(
        self,
        prev_stage: PipelineStage | None = None,
        prev_results: Any = None,
        layout_plot_pars: dict[str, Any] | None = None,
        verbosity: int = 1,
        **kwargs: Any,
    ) -> tuple[bool, np.ndarray]:
        """Run all random-subset optimizer steps."""
        del prev_stage, kwargs
        layout_xy = self._pipeline.read_layout(prev_results).copy()
        self._ensure_state_count(layout_xy)
        rng = np.random.default_rng(self.seed)

        def _run_steps() -> None:
            nonlocal layout_xy
            for step in range(self.n_steps):
                turbine_indices, state_indices = self._sample_subsets(rng)
                if verbosity > 0:
                    print(
                        f"{self.name}: Step {step + 1}/{self.n_steps}, "
                        f"turbines {turbine_indices.tolist()}, "
                        f"states {state_indices.tolist()}"
                    )
                accepted, candidate = self._optimize_step(
                    layout_xy,
                    turbine_indices,
                    state_indices,
                    verbosity,
                )
                if accepted:
                    layout_xy = candidate
                    if self.write_step_results:
                        self._write_step(
                            layout_xy,
                            step,
                            layout_plot_pars,
                            verbosity,
                        )
                elif verbosity > 0:
                    print(f"{self.name}: Step {step + 1} did not improve the layout")

        run_with_engine(_run_steps)
        success = layout_xy.shape == (self._pipeline.n_turbines, 2) and np.all(
            np.isfinite(layout_xy)
        )
        return bool(success), layout_xy