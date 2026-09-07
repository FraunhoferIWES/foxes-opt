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
    algo_pars
        The parameters for the foxes algorithm
    n_turbines
        The number of turbines in the wind farm
    turbine_models
        The turbine models
    mbook
        The model book
    farm_pars
        Additional parameters for the wind farm
    states
        The states to optimize the layout for


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
        base_dir
            Base directory for the self.
        algo_pars
            The parameters for the foxes algorithm
        n_turbines
            The number of turbines in the wind farm
        turbine_models
            The turbine models
        farm_boundary
            The wind farm boundary
        states
            The states to optimize the layout for
        mbook
            The model book to be used
        farm_pars
            Additional parameters for the wind farm
        name
            The name of the pipeline

        kwargs
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
    
    def get_algo(
        self,
        layout_xy: np.ndarray | None = None, 
        farm: WindFarm | None = None,
        states: States | None = None,
        initialize: bool = True,
        force: bool = False,
        verbosity: int = 0,
        **kwargs,
    ):
        """
        Create an algorithm instance

        Parameters
        ----------
        layout_xy
            The layout coordinates of the turbines, shape (n_turbines, 2)
        farm
            The wind farm instance, if None a new one will be created
        states
            The states to optimize the layout for, if None the states of the pipeline will be used
        initialize
            Whether to initialize the algorithm after creation
        force
            Whether to force re-initialization of the algorithm
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Additional keyword arguments for the algorithm creation

        Returns
        -------
        algo
            The algorithm instance

        """
        if farm is None:
            farm = WindFarm(boundary=self.farm_boundary, **self.farm_pars)
            assert layout_xy is not None, f"{self.name}: layout_xy must be provided if farm is None"
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
        if kwargs is not None:
            pars.update(kwargs)
        pars.setdefault("verbosity", verbosity)
        algo = Algorithm.new(
            farm=farm,
            states=states if states is not None else self.states,
            mbook=self.mbook,
            **pars,
        )

        if force or initialize:
            algo.initialize(force=force)

        return algo

    def run_foxes(
        self, 
        layout_xy: np.ndarray | None = None, 
        algo: Algorithm | None = None,
        farm: WindFarm | None = None,
        states: States | None = None,
        algo_pars: dict[str, Any] | None = None,
        initialize: bool = False,
        force: bool = True,
        verbosity: int = 0,
        **kwargs,
    ) -> tuple[Algorithm, Any]:
        """
        Run the foxes algorithm.

        Parameters
        ----------
        layout_xy
            The layout coordinates of the turbines, shape (n_turbines, 2)
        algo
            The foxes algorithm instance, if None a new one will be created
        farm
            The wind farm instance, if None a new one will be created
        states
            The states to optimize the layout for, if None the states of the pipeline will be used
        algo_pars
            Additional parameters for the foxes algorithm
        verbosity
            The verbosity level, 0 = silent
        force
            Whether to force the initialization of the algorithm
        initialize
            Whether to initialize the algorithm after creation
        kwargs
            Additional keyword arguments for the calc_farm method of the foxes algorithm

        Returns
        -------
        algo
            The foxes algorithm instance
        farm_results
            The results of the wind farm simulation

        """

        if algo is None:
            apars = algo_pars if algo_pars is not None else {}
            algo = self.get_algo(
                layout_xy=layout_xy,
                farm=farm,
                states=states,
                verbosity=verbosity,
                initialize=initialize,
                force=force,
                **apars,
            )

        results = run_with_engine(algo.calc_farm, **kwargs)

        return algo, results

    def read_layout(self, results: Any) -> np.ndarray:
        """
        Read the layout coordinates from the results.

        Parameters
        ----------
        results
            The results of the pipeline

        Returns
        -------
        layout_xy
            The layout coordinates of the turbines, shape (n_turbines, 2)

        """
        if isinstance(results, tuple):
            layout_xy = results[0]
        else:
            layout_xy = results

        if isinstance(layout_xy, np.ndarray):
            if layout_xy.shape == (self.n_turbines, 2):
                return layout_xy
            else:
                raise ValueError(f"Invalid layout coordinates shape, got {layout_xy.shape}, expected {(self.n_turbines, 2)}.")
        else:
            raise ValueError(f"Invalid results format for reading layout coordinates, got {type(layout_xy)}.")

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
        start_stage
            The stage index to start from
        end_stage
            The stage index to end at, default None (run all stages)
        finalize
            Whether to finalize the pipeline after running, default True
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        success
            Whether all stages were successful
        results
            The results of the pipeline, containing the algorithm and farm
            results when successful, otherwise the layout coordinates and no
            farm result.

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
            layout_xy = self.read_layout(results)
            algo, farm_results = self.run_foxes(layout_xy, force=True, verbosity=verbosity - 1)
            results = (algo, farm_results)
            if verbosity > 0:
                print(
                    f"Mean ambient REWS: {farm_results[FV.AMB_REWS].mean().values:.8f} m/s "
                )
        else:
            results = (results, None)

        return success, results
