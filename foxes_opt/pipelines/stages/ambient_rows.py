import numpy as np
from tqdm.autonotebook import tqdm
from scipy.interpolate import RegularGridInterpolator

from iwopy.core import PipelineStage

from foxes import WindFarm, Turbine
from foxes.core import Algorithm
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
        The stepsize_ortho orthogonal to main wind direction.
    stepsize_wd: float, optional
        The stepsize along the main wind direction.
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
        stepsize_wd=None,
        name="ambient_rows",
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        stepsize_ortho: float
            The stepsize_ortho orthogonal to main wind direction.
        mean_flow_states: foxes.input.states.SingleStateField
            The mean field flow states
        mean_flow_var2ncvar: dict
            Mapping from mean flow variable names to NetCDF variable names.
        stepsize_wd: float, optional
            The stepsize along the main wind direction.
        name: str, optional
            The name of the stage.

        kwargs: dict
             Additional keyword arguments for the stage.

        """
        super().__init__(name=name, **kwargs)
        self.stepsize_ortho = stepsize_ortho
        self.stepsize_wd = stepsize_wd
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

        # create algorithm object:
        farm = WindFarm(boundary=pipeline.farm_boundary, **pipeline.farm_pars)
        p_min = farm.boundary.p_min()
        p_max = farm.boundary.p_max()
        xy = np.linspace(p_min, p_max, pipeline.n_turbines)
        for i in range(pipeline.n_turbines):
            farm.add_turbine(
                Turbine(
                    xy=xy[i],
                    turbine_models=pipeline.turbine_models,
                    index=i,
                ),
                verbosity=verbosity - 1,
            )
        pars = pipeline.algo_pars.copy()
        pars.setdefault("verbosity", verbosity - 1)
        self._algo = Algorithm.new(
            farm=farm,
            states=self.mean_flow_states,
            mbook=pipeline.mbook,
            **pars,
        )
        self._algo.initialize()

        # get data:
        assert isinstance(self.mean_flow_states, SingleStateField), (
            f"{self.name}: mean_flow_states must be a SingleStateField, "
            f"got {type(self.mean_flow_states)}"
        )
        data = self._algo.loaded_data["extra_data"][self.mean_flow_states.DATA]

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
        if verbosity > 0:
            pbar = tqdm(
                total=self._n_turbines, desc=f"{self.name}: Running mean_ambient stage"
            )
        else:
            pbar = None

        # add turbines subsequently:

        pts = np.full((self._n_turbines, 2), np.nan, dtype=config.dtype_double)
        i = 0

        def _new_turbine(p):
            nonlocal i
            pts[i] = p
            i += 1
            if pbar is not None:
                pbar.update(1)

        def _next_u(u=-1):
            if u is None:
                return None
            while u + 1 < self._n_points:
                u = u + 1
                o = self._porder[u]
                if self._pvalid[o]:
                    d = np.linalg.norm(pts[:i] - self._points[None, o], axis=1)
                    if np.all(d >= self.stepsize_ortho):
                        break
            return u if u < self._n_points else None

        def _walk(p, n, dir, stepsize, res_p, res_ws, ws_next, cond=None):
            if i < self._n_turbines:
                q = p + stepsize * dir * n
                if self._farm_boundary.points_inside(q[None, :])[0] and (
                    cond is None or cond(q)
                ):
                    r = self._interpolator(q[None, :])[0]
                    ws = r[0]
                    if ws_next is None or ws >= ws_next:
                        d = np.linalg.norm(pts[:i] - q[None, :], axis=1)
                        if np.all(d >= stepsize):
                            n = np.array([-r[2], r[1]])
                            res_p.append(q)
                            res_ws.append(ws)
                            _walk(
                                q, n, dir, stepsize, res_p, res_ws, ws_next, cond=cond
                            )

        u = _next_u()
        while i < self._n_turbines:
            # get grid point data:
            if u is not None:
                o = self._porder[u]
                p = self._points[o]
                ws = self._ws[o]
                uv = wd2uv(self._wd[o])
                n = np.array([-uv[1], uv[0]])

            # find better point in up/donwstream direction if possible:
            if self.stepsize_wd is not None and self.stepsize_wd > 0:

                def cond(q):
                    return np.all(
                        np.linalg.norm(self._points - q[None, :], axis=1)
                        >= self.stepsize_wd
                    )

                res_p = [p]
                res_ws = [ws]
                _walk(
                    p, uv, 1, self.stepsize_wd, res_p, res_ws, ws_next=None, cond=cond
                )
                _walk(
                    p, uv, -1, self.stepsize_wd, res_p, res_ws, ws_next=None, cond=cond
                )
                if len(res_p) > 1:
                    j = np.argmax(res_ws)
                    p = res_p[j]
                    uv = self._interpolator(p[None, :])[0, 1:3]
                    n = np.array([-uv[1], uv[0]])

            # set first point in row:
            _new_turbine(p)

            # find next grid point:
            u = _next_u(u)
            ws_next = self._ws[self._porder[u]] if u is not None else None

            # add points in orthogonal direction:
            res_p_a = []
            res_ws_a = []
            _walk(p, n, 1, self.stepsize_ortho, res_p_a, res_ws_a, ws_next)
            res_p_b = []
            res_ws_b = []
            _walk(p, n, -1, self.stepsize_ortho, res_p_b, res_ws_b, ws_next)
            while len(res_p_a) > 0 or len(res_p_b) > 0:
                if len(res_ws_a) > 0 and (
                    len(res_ws_b) == 0 or res_ws_a[0] >= res_ws_b[0]
                ):
                    _new_turbine(res_p_a.pop(0))
                    res_ws_a.pop(0)
                elif len(res_ws_b) > 0:
                    _new_turbine(res_p_b.pop(0))
                    res_ws_b.pop(0)

        if pbar is not None:
            pbar.close()

        """
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(pts[:, 0], pts[:, 1], s=1)
        plt.xlabel("X coordinate (m)")
        plt.ylabel("Y coordinate (m)")
        plt.title("Turbine Positions in Ambient Rows")
        plt.axis("equal")
        plt.show()
        quit()
        """

        success = not np.any(np.isnan(pts))

        return success, pts
