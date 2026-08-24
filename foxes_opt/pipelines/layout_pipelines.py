from typing import Any

import numpy as np
from iwopy import Pipeline

from foxes import ModelBook, WindFarm, Turbine
from foxes.core import Algorithm, States, run_with_engine
from foxes.utils.geom2d import AreaGeometry
import foxes.variables as FV


class LayoutPipeline(Pipeline):
    """
    Pipeline for layout generation.

    Attributes
    ----------
    algo_pars: dict
        The parameters for the foxes algorithm
    n_turbines: int
        The number of turbines in the wind farm
    turbine_models: list of str
        The turbine models
    mbook: foxes.ModelBook
        The model book
    farm_pars: dict, optional
        Additional parameters for the wind farm
    states: foxes.core.States, optional
        The states to optimize the layout for

    :group: opt.pipelines

    """

    def __init__(
        self,
        base_dir: str,
        algo_pars: dict[str, Any],
        n_turbines: int,
        turbine_models: list[str],
        farm_boundary: AreaGeometry | None,
        states: States | None = None,
        mbook: ModelBook | None = None,
        farm_pars: dict[str, Any] | None = None,
        name: str = "layout_pipeline",
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        base_dir: str
            Base directory for the self.
        algo_pars: dict
            The parameters for the foxes algorithm
        n_turbines: int
            The number of turbines in the wind farm
        turbine_models: list of str
            The turbine models
        farm_boundary: foxes.utils.geom2d.AreaGeometry
            The wind farm boundary
        states: foxes.core.States, optional
            The states to optimize the layout for
        mbook: foxes.ModelBook, optional
            The model book to be used
        farm_pars: dict, optional
            Additional parameters for the wind farm
        name: str, optional
            The name of the pipeline

        kwargs: dict
            Additional keyword arguments for the self.

        """
        super().__init__(base_dir, name=name, **kwargs)
        self.algo_pars = algo_pars
        self.n_turbines = n_turbines
        self.states = states
        self.farm_boundary = farm_boundary
        self.turbine_models = turbine_models
        self.mbook = mbook if mbook is not None else ModelBook()
        self.farm_pars = {} if farm_pars is None else farm_pars

    def _run_foxes(
        self, layout_xy: np.ndarray, verbosity: int = 0
    ) -> tuple[Algorithm, Any]:
        """
        Run the foxes algorithm.

        Parameters
        ----------
        layout_xy: np.ndarray
            The layout coordinates of the turbines, shape (n_turbines, 2)
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        algo: foxes.core.Algorithm
            The foxes algorithm instance
        farm_results: xarray.Dataset
            The results of the wind farm simulation

        """

        farm = WindFarm(boundary=self.farm_boundary, **self.farm_pars)

        for i in range(self.n_turbines):
            farm.add_turbine(
                Turbine(
                    xy=layout_xy[i],
                    turbine_models=self.turbine_models,
                    index=i,
                ),
                verbosity=verbosity,
            )
        pars = self.algo_pars.copy()
        pars.setdefault("verbosity", verbosity)
        algo = Algorithm.new(
            farm=farm,
            states=self.states,
            mbook=self.mbook,
            **pars,
        )
        algo.initialize(force=True)

        return algo, run_with_engine(algo.calc_farm)

    def run(
        self,
        start_stage: int = 0,
        end_stage: int | None = None,
        finalize: bool = True,
        verbosity: int = 1,
    ) -> tuple[bool, tuple[Any, Any]]:
        """
        Run the pipeline.

        Parameters
        ----------
        start_stage: int
            The stage index to start from
        end_stage: int, optional
            The stage index to end at, default None (run all stages)
        finalize: bool
            Whether to finalize the pipeline after running, default True
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        success: bool
            Whether all stages were successful
        results: tuple
            The results of the pipeline, a tuple of (algo, farm_results)
            if successful, otherwise (layout_xy, None)

        """
        success, results = super().run(
            start_stage=start_stage,
            end_stage=end_stage,
            finalize=finalize,
            verbosity=verbosity,
        )

        if verbosity > 0:
            print("Pipeline finished.")
            if success:
                print("All stages were successful.")
            else:
                print("Some stages failed.")

        if (
            self.states is not None
            and results is not None
            and results.shape == (self.n_turbines, 2)
            and not np.any(np.isnan(results))
        ):
            algo, farm_results = self._run_foxes(results, verbosity=verbosity - 1)
            results = (algo, farm_results)
            if verbosity > 0:
                print(
                    f"Mean ambient REWS: {farm_results[FV.AMB_REWS].mean().values:.8f} m/s "
                )
        else:
            results = (results, None)

        return success, results
