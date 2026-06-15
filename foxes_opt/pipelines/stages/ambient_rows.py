import numpy as np
from tqdm.autonotebook import tqdm
from scipy.interpolate import RegularGridInterpolator

from iwopy.core import PipelineStage

from foxes.input.states import SingleStateField
from foxes import config
from foxes.utils.geom2d import ClosedPolygon
from foxes.utils import wd2uv
import foxes.variables as FV


class AmbientRowsStage(PipelineStage):
    """
    Pipeline stage for ambient rows.

    Attributes
    ----------
    stepsize_ortho: float
        The orthogonal stepsize orthogonal to main wind direction.
    mean_flow_states: foxes.input.input.states.SingleStateField
        The mean field flow states
    mean_flow_var2ncvar: dict
        Mapping from mean flow variable names to NetCDF variable names.

    :group: opt.pipelines

    """

    def __init__(
        self,
        stepsize_ortho,
        mean_flow_states,
        mean_flow_var2ncvar={},
        name="ambient_rows",
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        stepsize_ortho: float
            The orthogonal stepsize orthogonal to main wind direction.
        mean_flow_states: foxes.input.states.SingleStateField
            The mean field flow states
        mean_flow_var2ncvar: dict
            Mapping from mean flow variable names to NetCDF variable names.
        name: str, optional
            The name of the stage.

        kwargs: dict
             Additional keyword arguments for the stage.

        """
        super().__init__(name=name, **kwargs)
        self.stepsize_ortho = stepsize_ortho
        self.mean_flow_states = mean_flow_states
        self.mean_flow_var2ncvar = mean_flow_var2ncvar

    def initialize(self, pipeline, verbosity=0):
        """
        Initialize the stage. This method is called before running the stage.

        Parameters
        ----------
        pipeline: Pipeline
            The pipeline this stage belongs to
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().initialize(pipeline, verbosity=verbosity)

        # load data:
        assert isinstance(self.mean_flow_states, SingleStateField), (
            f"{self.name}: mean_flow_states must be a SingleStateField, "
            f"got {type(self.mean_flow_states)}"
        )
        if self.mean_flow_states.data is None:
            self.mean_flow_states.load_data(verbosity=verbosity)
        data = self.mean_flow_states.data

        # check variables:
        mean_ws_var = self.mean_flow_var2ncvar.get(FV.MEAN_WS, FV.MEAN_WS)
        if mean_ws_var not in data:
            raise KeyError(
                f"{self.name}: Mean wind speed variable '{mean_ws_var}' not found in "
                f"mean flow states data, got {list(data.keys())}"
            )
        main_wd_var = self.mean_flow_var2ncvar.get(FV.MAIN_WD, FV.MAIN_WD)
        if main_wd_var not in data:
            raise KeyError(
                f"{self.name}: Main wind direction variable '{main_wd_var}' not found in "
                f"mean flow states data, got {list(data.keys())}"
            )
        x_var = self.mean_flow_var2ncvar.get(FV.X, FV.X)
        y_var = self.mean_flow_var2ncvar.get(FV.Y, FV.Y)
        if x_var not in data or y_var not in data:
            raise KeyError(
                f"{self.name}: Missing '{x_var}' or '{y_var}' coordinates, got {list(data.keys())}"
            )

        # if no farm boundary is given, use the whole grid area:
        self._farm_boundary = pipeline.farm_boundary
        if self._farm_boundary is None:
            x_min = data.coords[x_var].min().item()
            x_max = data.coords[x_var].max().item()
            y_min = data.coords[y_var].min().item()
            y_max = data.coords[y_var].max().item()
            self._farm_boundary = ClosedPolygon(
                points=np.array(
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                )
            )
            if verbosity > 1:
                print(
                    f"{self.name}: No farm boundary given, using grid area: "
                    f"x: {x_min:.2f} - {x_max:.2f} m, y: {y_min:.2f} - {y_max:.2f} m"
                )

        # sort mean wind speeds on grid:
        ws_dims = data[mean_ws_var].dims
        if ws_dims != (x_var, y_var):
            raise NotImplementedError(
                f"{self.name}: Mean wind speed variable '{mean_ws_var}' must have dimensions "
                f"('{x_var}', '{y_var}'), got {ws_dims}"
            )
        ws = data[mean_ws_var].values
        self._nx, self._ny = ws.shape
        self._n_points = self._nx * self._ny
        if verbosity > 1:
            print(
                f"{self.name}: Grid dimensions: {self._nx} x {self._ny}, grid points: {self._n_points}"
            )
        self._ws = ws.reshape(self._n_points)
        self._porder = np.argsort(self._ws)[::-1]
        self._points = np.stack(
            np.meshgrid(
                data.coords[x_var].values,
                data.coords[y_var].values,
                indexing="ij",
            ),
            axis=-1,
        ).reshape(self._n_points, 2)
        self._pvalid = self._farm_boundary.points_inside(self._points)

        # prepare main wind direction data:
        wd_dims = data[main_wd_var].dims
        if wd_dims != (x_var, y_var):
            raise NotImplementedError(
                f"{self.name}: Main wind direction variable '{main_wd_var}' must have dimensions "
                f"('{x_var}', '{y_var}'), got {wd_dims}"
            )
        self._wd = data[main_wd_var].values.reshape(self._n_points)

        if verbosity > 1:
            print(
                f"{self.name}: Mean wind speeds range: {self._ws[self._porder[-1]]:.2f} - {self._ws[self._porder[0]]:.2f} m/s"
            )

        # setup interpolator:
        uv = wd2uv(data[main_wd_var].values)
        data = np.stack(
            (
                data[mean_ws_var].values,
                uv[..., 0],
                uv[..., 1],
            ),
            axis=-1,
        )
        del uv
        self._interpolator = RegularGridInterpolator(
            (
                self._points[:, 0].reshape(self._nx, self._ny)[:, 0],
                self._points[:, 1].reshape(self._nx, self._ny)[0, :],
            ),
            data,
            bounds_error=False,
            fill_value=None,
        )

        # initial data:
        self._n_turbines = pipeline.n_turbines
        self._xy = np.full((self._n_turbines, 2), np.nan, dtype=config.dtype_double)

    def run(self, prev_stage=None, prev_results=None, verbosity=1):
        """
        Run the pipeline stage.

        Parameters
        ----------
        prev_stage: PipelineStage, optional
            The previous stage
        prev_results: object, optional
            The results from the previous stage
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        success: bool
            Whether the stage was successful
        results: object
            The stage results

        """

        # prepare:
        results = None
        if verbosity > 0:
            pbar = tqdm(
                total=self._n_turbines, desc=f"{self.name}: Running mean_ambient stage"
            )
        else:
            pbar = None

        def _find_next(u):
            o = self._porder[u]
            if self._pvalid[o]:
                return o
            else:
                # p = self._points[o] + wd2uv(self._wd[o]) * self.stepsize_ortho
                raise NotImplementedError("TODO")

        # add turbines sequentially:
        for i in range(self._n_turbines):
            o = _find_next(0)
            if o is not None:
                self._xy[i] = self._points[o]
                self._pvalid[o] = False
            else:
                raise RuntimeError(f"{self.name}: No valid point found for turbine {i}")

            if pbar is not None:
                pbar.update()
        if pbar is not None:
            pbar.close()

        return True, results
