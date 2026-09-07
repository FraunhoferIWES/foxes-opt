from __future__ import annotations

import numpy as np
from tqdm.autonotebook import tqdm
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
from typing import Any, TYPE_CHECKING

from iwopy import Pipeline
from iwopy.core import PipelineStage

from foxes import WindFarm, Turbine
from foxes.core import Algorithm, run_with_engine
from foxes.algorithms.downwind.models import PopulationModel, PopulationStates
from foxes.input.states import SingleStateField
from foxes import config
from foxes.utils.geom2d import ClosedPolygon
from foxes.utils import wd2uv
from foxes.models.farm_controllers import OpFlagController
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core import States

class GreedyTurbineSweepStage(PipelineStage):
    """
    Pipeline stage for a sweep over greedy turbines.

    Attributes
    ----------
    max_move_dist
        The maximum distance a turbine can move during the sweep.
    stepsize
        The step size for moving the turbines during the sweep.
    n_sweeps
        The number of sweeps to perform.
    max_wake_length
        The maximum length of the wake.
    flow_states
        The ambient states
    ambient
        Whether the stage is in ambient mode (no wakes).

    """

    def __init__(
        self,
        max_move_dist: float,
        stepsize: float,
        n_sweeps: int = 1,
        max_wake_length: float = 5000.0,
        flow_states: States | None = None,
        ambient: bool = False,
        name: str = "greedy_turbine_sweep",
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        max_move_dist
            The maximum distance a turbine can move during the sweep.
        stepsize
            The step size for moving the turbines during the sweep.
        n_sweeps
            The number of sweeps to perform.
        max_wake_length
            The maximum length of the wake.
        flow_states
            The ambient states
        ambient
            Whether the stage is in ambient mode (no wakes).
        name
            The name of the stage.

        kwargs
             Additional keyword arguments for the stage.

        """
        super().__init__(name=name, **kwargs)
        self.max_move_dist = max_move_dist
        self.stepsize = stepsize
        self.n_sweeps = n_sweeps
        self.max_wake_length = max_wake_length
        self.flow_states = flow_states
        self.ambient = ambient

    def initialize(self, pipeline: Pipeline, verbosity: int = 0) -> None:
        """
        Initialize the stage. This method is called before running the stage.

        Parameters
        ----------
        pipeline
            The pipeline this stage belongs to
        verbosity
            The verbosity level, 0 = silent

        """
        super().initialize(pipeline, verbosity=verbosity)

        # set pipeline:
        self._pipeline = pipeline

        # update flow states:
        if self.flow_states is None:
            self.flow_states = pipeline.states

        if verbosity > 0:
            print(f"{self.name} parameters:")
            print(f"  stepsize: {self.stepsize} m")
            print(f"  max move dist: {self.max_move_dist} m")
            print(f"  max wake length: {self.max_wake_length} m")
            print(f"  flow states: {self.flow_states}")
            print(f"  n sweeps: {self.n_sweeps}")

    def run(
        self,
        prev_stage: PipelineStage | None = None,
        prev_results: Any = None,
        verbosity: int = 1,
    ) -> tuple[bool, np.ndarray]:
        """
        Run the pipeline stage.

        Parameters
        ----------
        prev_stage
            The previous stage
        prev_results
            The results from the previous stage
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        success
            Whether the stage was successful
        results
            The stage results

        """

        # prepare:
        pipeline = self._pipeline
        layout_xy = pipeline.read_layout(prev_results)
        n_turbines = pipeline.n_turbines

        # prepare xy delta grid:
        dx0 = np.arange(0.0, self.max_move_dist + 1e-6, self.stepsize)
        dx1 = np.arange(-self.stepsize, -self.max_move_dist - 1e-6, -self.stepsize)
        dx = np.concatenate([np.flip(dx1), dx0])
        n_dx = len(dx)
        assert n_dx > 1, f"{self.name}: step array is empty, check stepsize={self.stepsize} and max_move_dist={self.max_move_dist}"
        dxy = np.zeros((n_dx, n_dx, 2), dtype=config.dtype_double)
        dxy[:, :, 0] = dx[:, None]
        dxy[:, :, 1] = dx[None, :]
        dxy = dxy[np.linalg.norm(dxy, axis=-1) <= self.max_move_dist]
        if verbosity > 0:
            print(f"{self.name}: Delta xy shape = {dxy.shape}")
        idx_p0 = np.where(np.all(dxy == np.zeros((1, 2)), axis=-1))[0][0]
        del dx0, dx1, dx

        if verbosity == 1:
            pbar = tqdm(total=n_turbines, leave=True)

        def _run_sweeps():

            for sweepi in range(self.n_sweeps):

                if verbosity == 1:
                    pbar.reset()
                    pbar.set_description(f"{self.name}: Sweep {sweepi}/{self.n_sweeps}")

                # turbine sweep:
                for i in range(n_turbines):
                    if verbosity > 1:
                        print(f"{self.name}: run {i} Entering turbine {i}")

                    # create wind farm with only nearby turbines, without current turbine:
                    p0 = layout_xy[i, :].copy()
                    layout_xy[i, :] = np.nan
                    tsel = np.where(np.linalg.norm(layout_xy - p0[None, :], axis=-1) <= self.max_wake_length)[0]
                    farm = WindFarm(name=f"subfarm_turbine_{i}")
                    for j in tsel:
                        if j != i:
                            farm.add_turbine(
                                Turbine(
                                    xy=layout_xy[j, :],
                                    name=f"T_{j}",
                                    turbine_models=pipeline.turbine_models,
                                ),
                                verbosity=0,
                            )
                    ambient = self.ambient or farm.n_turbines < 2

                    # create algorithm:
                    algo = pipeline.get_algo(
                        farm=farm,
                        states=self.flow_states,
                        initialize=True,
                        force=True,
                        verbosity=0,
                    )

                    # calculate farm results:
                    farm_results = algo.calc_farm(ambient=ambient)
                    print("HERE", farm_results)

                    # calculate results at candidate points:
                    H = farm_results[FV.H].mean().values
                    xyz = np.concatenate([
                        p0[None, :] + dxy, 
                        np.broadcast_to(H, dxy[:, :1].shape),
                    ], axis=-1)
                    point_results = algo.calc_points(
                        farm_results,
                        points=np.broadcast_to(xyz[None, :, :], (algo.n_states,) + xyz.shape),
                        outputs=[FV.WS],
                    )
                    weights = farm_results[FV.WEIGHT]
                    if weights.dims == (FC.STATE, FC.TURBINE):
                        weights = weights.values
                    elif weights.dims == (FC.STATE,):
                        weights = weights.values[:, None]
                    else:
                        raise ValueError(f"{self.name}: Unexpected weights dimensions for run at turbine {i}: weights.dims = {weights.dims}")
                    point_results = np.einsum('sp,sp->p', point_results[FV.WS].values, weights)

                    # find best point:
                    best_idx = np.argmax(point_results)
                    new_p = xyz[best_idx, :2]
                    if verbosity > 1:
                        wsa = point_results[idx_p0]
                        wsb = point_results[best_idx]
                        prcnt = 100 * (wsb - wsa) / wsa
                        prcnt = f"+{prcnt:.2f}%" if prcnt > 0 else f"{prcnt:.2f}%"
                        print(f"--> Moving turbine {i} from {p0} to {new_p}, ws: {wsa:.2f} --> {wsb:.2f} m/s, {prcnt}")
                    layout_xy[i, :] = xyz[best_idx, :2]

                    if verbosity == 1:
                        pbar.update(1)
                
        run_with_engine(_run_sweeps)
        if verbosity == 1:
            pbar.close()

        success = not np.any(np.isnan(layout_xy))
        if verbosity > 0:
            if success:
                print(f"{self.name}: Successfully placed all turbines.")
            else:
                print(f"{self.name}: Failed to place all turbines.")
                if np.any(np.isnan(layout_xy)):
                    print(
                        f"{self.name}: Some turbine positions are NaN: {np.isnan(layout_xy).sum()} turbines."
                    )

        return success, layout_xy
