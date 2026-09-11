from __future__ import annotations

from typing import Any

import numpy as np
from foxes.core import SubsetStates, run_with_engine
from iwopy import Pipeline
from iwopy.core import PipelineStage

from .layout_optimizer import LayoutOptimizerStage


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
        n_subset_turbines: int | None,
        n_subset_states: int | None,
        n_steps: int,
        *args: Any,
        seed: int | None = None,
        write_step_results: bool = False,
        name: str = "random_subset",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        n_subset_turbines
            Number of movable turbines in each optimization step, or ``None``
            to optimize all turbines.
        n_subset_states
            Number of active states in each optimization step, or ``None`` to
            use all states.
        n_steps
            Number of successive random-subset optimizations.
        args
            Positional arguments forwarded to ``LayoutOptimizerStage``.
        seed
            Random seed. The generator is reset for every stage run.
        write_step_results
            Write full-state pipeline outputs after every accepted step.
        name
            Stage name.
        kwargs
            Keyword arguments forwarded to ``LayoutOptimizerStage``.
        """
        base_kwargs = kwargs.copy()
        base_kwargs["name"] = name
        super().__init__(*args, **base_kwargs)
        self.n_subset_turbines = n_subset_turbines
        self.n_subset_states = n_subset_states
        self.n_steps = n_steps
        self.seed = seed
        self.write_step_results = write_step_results

    def initialize(self, pipeline: Pipeline, verbosity: int = 0) -> None:
        """
        Initialize and validate random subset settings.

        Parameters
        ----------
        pipeline
            The pipeline hosting this random-subset stage.
        verbosity
            Verbosity level used during initialization.
        """
        super().initialize(pipeline, verbosity=verbosity)
        if self.n_subset_turbines is not None and not (
            0 < self.n_subset_turbines <= pipeline.n_turbines
        ):
            raise ValueError(
                f"{self.name}: n_subset_turbines must be in [1, {pipeline.n_turbines}], got {self.n_subset_turbines}"
            )
        self._n_flow_states = self._flow_states.size()
        if self.n_subset_states is not None and self.n_subset_states <= 0:
            raise ValueError(f"{self.name}: n_subset_states must be positive")
        if (
            self.n_subset_states is not None
            and self._n_flow_states
            and self.n_subset_states > self._n_flow_states
        ):
            raise ValueError(
                f"{self.name}: n_subset_states must be in [1, {self._n_flow_states}], got {self.n_subset_states}"
            )
        if self.n_steps <= 0:
            raise ValueError(f"{self.name}: n_steps must be positive")

    def _sample_subsets(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Draw one turbine subset and one state subset, when configured.

        Parameters
        ----------
        rng
            Random number generator used to sample the subset indices.

        Returns
        -------
        tuple[np.ndarray | None, np.ndarray | None]
            The sampled turbine indices and state indices, or ``None`` for the
            full set.
        """
        if self.n_subset_turbines is None:
            turbine_indices = None
        else:
            turbine_indices = rng.choice(
                self._pipeline.n_turbines,
                size=self.n_subset_turbines,
                replace=False,
            )
        if self.n_subset_states is None:
            state_indices = None
        else:
            state_indices = rng.choice(
                self._n_flow_states,
                size=self.n_subset_states,
                replace=False,
            )
        return turbine_indices, state_indices

    def _ensure_state_count(self, layout_xy: np.ndarray) -> None:
        """
        Initialize lazy states once to determine their full size.

        Parameters
        ----------
        layout_xy
            Current turbine layout used to construct the algorithm if needed.
        """
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
        if (
            self.n_subset_states is not None
            and self.n_subset_states > self._n_flow_states
        ):
            raise ValueError(
                f"{self.name}: n_subset_states must be in [1, {self._n_flow_states}], got {self.n_subset_states}"
            )

    def _optimize_step(
        self,
        layout_xy: np.ndarray,
        turbine_indices: np.ndarray | None,
        state_indices: np.ndarray | None,
        verbosity: int,
    ) -> tuple[bool, np.ndarray]:
        """
        Run the selected optimizer for one sampled layout problem.

        Parameters
        ----------
        layout_xy
            Current turbine layout.
        turbine_indices
            Indices of the turbines to optimize in this step, or ``None`` for
            all turbines.
        state_indices
            Indices of the states to optimize in this step, or ``None`` for all
            states.
        verbosity
            Verbosity level for the optimization run.

        Returns
        -------
        tuple[bool, np.ndarray]
            ``True`` together with the accepted candidate layout when the
            optimization succeeds; otherwise ``False`` and the previous layout.
        """
        results, candidate = self._run_layout_optimizer(
            layout_xy,
            states=(
                self._flow_states
                if state_indices is None
                else SubsetStates(self._flow_states, state_indices)
            ),
            sel_turbines=None if turbine_indices is None else turbine_indices.tolist(),
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
        """
        Write full-state outputs for an accepted optimization step.

        Parameters
        ----------
        layout_xy
            Optimized turbine layout to export.
        step
            Zero-based step index used in the output table and file names.
        layout_plot_pars
            Optional plotting configuration overrides.
        verbosity
            Verbosity level used when running the full-state evaluation.
        """
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
        """
        Run all random-subset optimizer steps.

        Parameters
        ----------
        prev_stage
            Previous pipeline stage, unused by this implementation.
        prev_results
            Results from the previous stage, used to read the current layout.
        layout_plot_pars
            Optional plot configuration forwarded to full-state outputs.
        verbosity
            Verbosity level used during subset optimization.
        kwargs
            Additional keyword arguments, unused by this stage.

        Returns
        -------
        tuple[bool, np.ndarray]
            ``True`` and the final layout after all accepted steps, or ``False``
            with a fallback layout if the operation did not produce a valid
            result.
        """
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
                        f"turbines {'all' if turbine_indices is None else turbine_indices.tolist()}, "
                        f"states {'all' if state_indices is None else state_indices.tolist()}"
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
