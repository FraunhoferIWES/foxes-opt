from pathlib import Path
from typing import Any

import foxes.constants as FC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from foxes import ModelBook, Turbine, WindFarm
from foxes.core import Algorithm, States, run_with_engine
from foxes.output import FarmLayoutOutput, FarmResultsEval, FlowPlots2D
from foxes.utils.geom2d import AreaGeometry
from iwopy import Pipeline, PipelineStage
from xarray import Dataset


class LayoutPipeline(Pipeline):
    """
    Pipeline for layout generation.
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
        reset_table: bool = False,
        name: str = "layout_pipeline",
        **kwargs: Any,
    ) -> None:
        """
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
        reset_table
            Whether to reset the table file if it exists
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
        self.reset_table = reset_table
        self.__table: pd.DataFrame | None = None

    @property
    def table_path(self) -> Path:
        """
        The path to the table file.

        Returns
        -------
        table_path
            The path to the table file.

        """
        return self.base_dir / f"{self.name}.csv"

    @property
    def table(self) -> pd.DataFrame:
        """
        The table of the pipeline.

        Returns
        -------
        table
            The table of the pipeline.

        """
        if self.__table is None:
            if self.table_path.exists() and not self.reset_table:
                print(f"{self.name}: Loading table from {self.table_path}")
                self.__table = pd.read_csv(self.table_path, index_col="index")
            else:
                if self.table_path.exists() and self.reset_table:
                    print(f"{self.name}: Resetting table at {self.table_path}")
                    self.table_path.unlink()
                else:
                    print(f"{self.name}: Creating empty table at {self.table_path}")
                self.__table = pd.DataFrame(
                    columns=[
                        "index",
                        "stage",
                        "success",
                        "n_turbines",
                        "capacity",
                        "efficiency",
                        "yield",
                    ]
                )
                self.__table.set_index("index", inplace=True)
                self.write_table()

        return self.__table

    def write_table(self) -> None:
        """
        Write the table to the file.

        """
        self.table.to_csv(self.table_path)

    def add_to_table(
        self,
        stage: PipelineStage | str,
        success: bool,
        algo: Algorithm,
        farm_results: Dataset,
    ) -> int:
        """
        Add the results of a stage to the table.

        Parameters
        ----------
        stage
            The stage or the name of the stage
        success
            Whether the stage was successful
        algo
            The foxes algorithm instance
        farm_results
            The results of the wind farm simulation

        Returns
        -------
        i
            The index of the new row in the table

        """
        if success:
            o = FarmResultsEval(farm_results, algo)
            Y = o.calc_farm_yield(annual=True)
            eff = o.calc_farm_efficiency()
            cap = o.calc_farm_capacity()
        else:
            Y = np.nan
            eff = np.nan
            cap = np.nan

        if isinstance(stage, PipelineStage):
            stage = stage.name

        i = len(self.table)
        new_row = pd.DataFrame(
            {
                "index": [i],
                "stage": [stage],
                "success": [success],
                "n_turbines": [farm_results.sizes[FC.TURBINE]],
                "capacity": [cap],
                "efficiency": [eff],
                "yield": [Y],
            },
        ).set_index("index")

        self.__table = pd.concat([self.table, new_row])
        self.write_table()

        return i

    def write_layout_plot(
        self,
        algo: Algorithm,
        farm_results: Dataset,
        title: str | None = None,
        table_index: int | None = None,
        figsize: tuple[float, float] = (12, 8),
        show_flow: bool = True,
        resolution: float = 50.0,
        alpha: float = 0.6,
        annotate: int = 0,
        final: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Write a plot of the layout and the farm results.

        Parameters
        ----------
        algo
            The foxes algorithm instance
        farm_results
            The results of the wind farm simulation
        plot_path
            The path to save the plot
        title
            The title of the plot
        figsize
            The size of the figure
        show_flow
            Whether to show the flow field in the plot
        resolution
            The resolution of the flow field plot
        alpha
            The alpha value for the flow field plot
        annotate
            The turbine annotation level
        final
            Whether this is the final layout plot

        """
        farm = algo.farm
        if final:
            plot_dir = self.base_dir
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / "layout_final.jpg"
            if table_index is None:
                table_index = len(self.table) - 1
            elif table_index != len(self.table) - 1:
                raise ValueError(
                    f"{self.name}: For final layout plot, table_index must be the last index of the table"
                )
        else:
            plot_dir = self.base_dir / "layouts" / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"layout_{table_index:04d}.jpg"

        if show_flow:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            ax = axs[0]
        else:
            fig, ax = plt.subplots(figsize=figsize)

        if title is None:
            assert table_index is not None, (
                f"{self.name}: Either title or table_index must be provided"
            )
            stage = self.table.loc[table_index, "stage"]
            N = self.table.loc[table_index, "n_turbines"]
            Y = self.table.loc[table_index, "yield"]
            eff = self.table.loc[table_index, "efficiency"]
            title = f"Layout {table_index}, {stage}: {N} turbines, yield = {Y:.2f} GWh, eff = {100 * eff:.1f} %"

        print(f"{self.name}: Creating layout plot {plot_path}")

        o = FarmLayoutOutput(farm, farm_results=farm_results)
        o.get_figure(
            fig=fig,
            ax=ax,
            annotate=annotate,
            title=None if show_flow else title,
            **kwargs,
        )

        if show_flow:
            fig.suptitle(title)

            p_min = farm.boundary.p_min()
            p_max = farm.boundary.p_max()
            o = FlowPlots2D(algo, farm_results)
            plot_data = run_with_engine(
                o.get_mean_data_xy,
                "WS",
                resolution=resolution,
                xmin=p_min[0],
                xmax=p_max[0],
                ymin=p_min[1],
                ymax=p_max[1],
            )

            fig = o.get_mean_fig_xy(plot_data, fig=fig, ax=axs[1])
            dpars = {"alpha": alpha, "zorder": 10, "p_min": p_min, "p_max": p_max}
            farm.boundary.add_to_figure(
                axs[1], fill_mode="outside_white", pars_distance=dpars
            )

        plt.savefig(plot_path, bbox_inches="tight")

    def write_layout_csv(
        self,
        algo: Algorithm,
        farm_results: Dataset,
        table_index: int | None = None,
        final: bool = False,
    ) -> None:
        """
        Write the layout coordinates to a CSV file.

        Parameters
        ----------
        algo
            The foxes algorithm instance
        farm_results
            The results of the wind farm simulation
        table_index
            The index of the row in the table to use for naming the CSV file
        final
            Whether this is the final layout CSV file

        """
        if final:
            csv_dir = self.base_dir
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / "layout_final.csv"
            if table_index is None:
                table_index = len(self.table) - 1
            elif table_index != len(self.table) - 1:
                raise ValueError(
                    f"{self.name}: For final layout CSV, table_index must be the last index of the table"
                )
        else:
            csv_dir = self.base_dir / "layouts"
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / f"layout_{table_index:04d}.csv"

        print(f"{self.name}: Creating layout CSV {csv_path}")

        o = FarmLayoutOutput(farm=algo.farm, farm_results=farm_results, algo=algo)
        o.write_csv(csv_path, verbosity=0)

    def get_algo(
        self,
        layout_xy: np.ndarray | None = None,
        farm: WindFarm | None = None,
        states: States | None = None,
        initialize: bool = True,
        force: bool = False,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Algorithm:
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
            assert layout_xy is not None, (
                f"{self.name}: layout_xy must be provided if farm is None"
            )
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
        **kwargs: Any,
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
                raise ValueError(
                    f"Invalid layout coordinates shape, got {layout_xy.shape}, expected {(self.n_turbines, 2)}."
                )
        else:
            raise ValueError(
                f"Invalid results format for reading layout coordinates, got {type(layout_xy)}."
            )

    def run(
        self,
        start_stage: int = 0,
        end_stage: int | None = None,
        finalize: bool = True,
        layout_plot_pars: dict[str, Any] | None = None,
        layout_plot_pars_final: dict[str, Any] | None = None,
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
        layout_plot_pars
            Additional parameters for the layout plot, default None
        layout_plot_pars_final
            Additional parameters for the final layout plot, default None
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
            layout_plot_pars=layout_plot_pars,
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
            algo, farm_results = self.run_foxes(
                layout_xy, force=True, verbosity=verbosity - 1
            )
            results = (algo, farm_results)

            i = self.add_to_table("final", success, algo, farm_results)
            self.write_layout_csv(algo, farm_results, table_index=i)

            pars: dict[str, Any] = {
                "figsize": (12, 8),
                "show_flow": False,
                "resolution": 50.0,
                "annotate": 0,
            }
            if layout_plot_pars is not None:
                pars.update(layout_plot_pars)
            if layout_plot_pars_final is not None:
                pars.update(layout_plot_pars_final)

            self.write_layout_plot(
                algo, farm_results, table_index=i, final=True, **pars
            )

        else:
            results = (results, None)

        return success, results
