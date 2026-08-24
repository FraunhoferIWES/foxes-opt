from typing import Any

import numpy as np
from tqdm.autonotebook import tqdm
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist

from iwopy import Pipeline
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
        stepsize_ortho: float,
        mean_flow_states: SingleStateField,
        mean_flow_var2ncvar: dict[str, str] | None = None,
        stepsize_wd: float | None = None,
        name: str = "ambient_rows",
        **kwargs: Any,
    ) -> None:
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
        self.mean_flow_var2ncvar = (
            {} if mean_flow_var2ncvar is None else mean_flow_var2ncvar
        )

    def initialize(self, pipeline: Pipeline, verbosity: int = 0) -> None:
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

        if verbosity > 0:
            print(f"{self.name}: stepsize_ortho: {self.stepsize_ortho} m")
            print(f"{self.name}: stepsize_wd: {self.stepsize_wd} m")

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
        self._points = np.stack(
            np.meshgrid(
                data.coords[x_var].values,
                data.coords[y_var].values,
                indexing="ij",
            ),
            axis=-1,
        ).reshape(self._n_points, 2)
        pvalid = self._farm_boundary.points_inside(self._points)

        # prepare main wind direction data:
        wd_dims = data[main_wd_var].dims
        if wd_dims != (x_var, y_var):
            raise NotImplementedError(
                f"{self.name}: Main wind direction variable '{main_wd_var}' must have dimensions "
                f"('{x_var}', '{y_var}'), got {wd_dims}"
            )
        self._wd = data[main_wd_var].values.reshape(self._n_points)

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
        self._interpolator: RegularGridInterpolator[
            tuple[
                np.ndarray[tuple[int, ...], np.dtype[Any]],
                np.ndarray[tuple[int, ...], np.dtype[Any]],
            ],
            np.ndarray[tuple[int, ...], np.dtype],
        ] = RegularGridInterpolator(
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

        # filter points outside farm boundary:
        self._points = self._points[pvalid]
        self._n_points = self._points.shape[0]
        self._ws = self._ws[pvalid]
        self._wd = self._wd[pvalid]
        self._porder: np.ndarray[tuple[int, ...], np.dtype[np.signedinteger]] = (
            np.argsort(self._ws)[::-1]
        )

        if verbosity > 0:
            print(
                f"{self.name}: Mean wind speeds range: {self._ws[self._porder[-1]]:.2f} - {self._ws[self._porder[0]]:.2f} m/s"
            )

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
            pbar: tqdm = tqdm(
                total=self._n_turbines, desc=f"{self.name}: Running mean_ambient stage"
            )
        else:
            pbar = None

        # add turbines subsequently:

        pts = np.full((self._n_turbines, 2), np.nan, dtype=config.dtype_double)
        i = 0

        def _new_turbine(p: np.ndarray) -> None:
            nonlocal i
            pts[i] = p
            i += 1
            if pbar is not None:
                pbar.update(1)

        def _is_near(
            p: np.ndarray,
            pts: np.ndarray,
            uv: np.ndarray | None = None,
            n: np.ndarray | None = None,
        ) -> np.bool_:
            d = pts - p[None, :]
            if self.stepsize_wd is not None:
                if uv is None:
                    uv = self._interpolator(p[None, :])[0, 1:3]
                if n is None:
                    n = np.array([-uv[1], uv[0]])
                return np.any(
                    (np.abs(d @ uv) < self.stepsize_wd - 1e-10)
                    & (np.abs(d @ n) < self.stepsize_ortho - 1e-10)
                )
            else:
                return np.any(np.linalg.norm(d, axis=1) < self.stepsize_ortho - 1e-10)

        skip_u = set()

        def _next_u(u: int = -1) -> None | int:
            # u = -1
            while True:
                u = u + 1
                if u in skip_u:
                    continue
                if u >= self._n_points:
                    return None
                o = self._porder[u]
                uv = wd2uv(self._wd[o])
                if _is_near(self._points[o], pts[:i], uv=uv):
                    skip_u.add(u)
                else:
                    return u

        def _walk(
            p: np.ndarray,
            n: np.ndarray,
            dir: int,
            stepsize: float,
            res_puvdir: list[tuple[np.ndarray, np.ndarray, int]],
            res_ws: list[float],
            ws_next: float | None,
            cnd_puvdir: list[tuple[np.ndarray, np.ndarray, int]] | None = None,
            cnd_ws: list[float] | None = None,
            cond: Any = None,
        ) -> None:
            cnd_puvdir_l = [] if cnd_puvdir is None else cnd_puvdir
            cnd_ws_l = [] if cnd_ws is None else cnd_ws
            if i < self._n_turbines:
                # search candidates:
                ws0 = None
                if len(cnd_puvdir_l):
                    pops = set()
                    for j in np.argsort(cnd_ws_l)[::-1]:
                        q0, uv0, __ = cnd_puvdir_l[j]
                        if _is_near(q0, pts[:i], uv=uv0):
                            pops.add(j)
                        elif (cond is None or cond(cnd_puvdir_l[j][0])) and (
                            ws_next is None or cnd_ws_l[j] >= ws_next
                        ):
                            q0, uv0, dir0 = cnd_puvdir_l[j]
                            ws0 = cnd_ws_l[j]
                            pops.add(j)
                            break
                    for j in sorted(pops, reverse=True):
                        cnd_puvdir_l.pop(j)
                        cnd_ws_l.pop(j)

                # walk in direction:
                q = p + stepsize * dir * n
                valid = self._farm_boundary.points_inside(q[None, :])[0] and (
                    cond is None or cond(q)
                )
                if valid:
                    r = self._interpolator(q[None, :])[0]
                    ws = r[0]
                    uv = r[1:3]

                # select candidate or walk:
                if ws0 is None and not valid:
                    return
                elif ws0 is None:
                    pass
                elif not valid or ws0 >= ws:
                    q = q0
                    ws = ws0
                    uv = uv0
                    dir = dir0
                else:
                    cnd_puvdir_l.append((q0, uv0, dir0))
                    cnd_ws_l.append(ws0)

                # check if better then next row point:
                if ws_next is None or ws >= ws_next:
                    n = np.array([-uv[1], uv[0]])
                    if not _is_near(q, pts[:i], uv=uv):
                        res_puvdir.append((q, uv, dir))
                        res_ws.append(ws)
                        _walk(
                            q,
                            n,
                            dir,
                            stepsize,
                            res_puvdir,
                            res_ws,
                            ws_next,
                            cnd_puvdir,
                            cnd_ws,
                            cond=cond,
                        )
                elif cnd_puvdir is not None and cnd_ws is not None:
                    cnd_puvdir_l.append((q, uv, dir))
                    cnd_ws_l.append(ws)

        u: int | None = 0
        cnd_puvdir: list[tuple[np.ndarray, np.ndarray, int]] = []
        cnd_ws: list[float] = []
        while i < self._n_turbines:
            if u is None:
                break
            # get grid point data:
            o = self._porder[u]
            p: np.ndarray = self._points[o]
            ws = self._ws[o]
            uv = wd2uv(self._wd[o])
            n = np.array([-uv[1], uv[0]])
            if _is_near(p, pts[:i], uv=uv):
                u = _next_u(u)
                if u is None:
                    break
                continue

            # find better point in up/donwstream direction if possible:
            new_p = False
            if self.stepsize_wd is not None and self.stepsize_wd > 0:

                def cond(q: np.ndarray) -> bool:
                    return not _is_near(q, self._points)

                res_puvdir = [(p, uv, 0)]
                res_ws = [ws]
                _walk(
                    p,
                    uv,
                    1,
                    self.stepsize_wd,
                    res_puvdir,
                    res_ws,
                    ws_next=None,
                    cond=cond,
                )
                _walk(
                    p,
                    uv,
                    -1,
                    self.stepsize_wd,
                    res_puvdir,
                    res_ws,
                    ws_next=None,
                    cond=cond,
                )
                if len(res_puvdir) > 1:
                    new_p = True
                    j = int(np.argmax(res_ws))
                    p, uv, _ = res_puvdir[j]
                    n = np.array([-uv[1], uv[0]])

            if not new_p:
                u = _next_u(u)
                ws = self._ws[self._porder[u]] if u is not None else None
                if u is None:
                    break

            # set first point in row:
            _new_turbine(p)

            # add points in orthogonal direction:
            res_puvdir_a: list[tuple[np.ndarray, np.ndarray, int]] = []
            res_ws_a: list[float] = []
            _walk(
                p,
                n,
                1,
                self.stepsize_ortho,
                res_puvdir_a,
                res_ws_a,
                ws,
                cnd_puvdir,
                cnd_ws,
            )
            res_puvdir_b: list[tuple[np.ndarray, np.ndarray, int]] = []
            res_ws_b: list[float] = []
            _walk(
                p,
                n,
                -1,
                self.stepsize_ortho,
                res_puvdir_b,
                res_ws_b,
                ws,
                cnd_puvdir,
                cnd_ws,
            )
            while i < self._n_turbines and (
                len(res_puvdir_a) > 0 or len(res_puvdir_b) > 0
            ):
                if len(res_ws_a) > 0 and (
                    len(res_ws_b) == 0 or res_ws_a[0] >= res_ws_b[0]
                ):
                    p0 = res_puvdir_a.pop(0)[0]
                    if not _is_near(p0, pts[:i]):
                        _new_turbine(p0)
                    res_ws_a.pop(0)
                elif len(res_ws_b) > 0:
                    p0 = res_puvdir_b.pop(0)[0]
                    if not _is_near(p0, pts[:i]):
                        _new_turbine(p0)
                    res_ws_b.pop(0)

        if pbar is not None:
            pbar.close()

        """ # Debugging: Plot turbine positions
        import matplotlib.pyplot as plt
        plt.figure()
        #plt.scatter(self._points[:, 0], self._points[:, 1], s=1, alpha=0.5, label="Grid Points")
        plt.scatter(pts[:, 0], pts[:, 1], s=1, label="Turbine Positions")
        #pts = self._points[[self._porder[u] for u in skip_u], :]
        #plt.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.5, label="Skipped Points")
        plt.xlabel("X coordinate (m)")
        plt.ylabel("Y coordinate (m)")
        plt.title("Turbine Positions in Ambient Rows")
        plt.axis("equal")
        #plt.legend()
        plt.show()
        """

        # check minimum distance between turbines:
        dmat: np.ndarray[tuple[int, ...], np.dtype[np.floating]] = cdist(pts, pts)
        np.fill_diagonal(dmat, np.inf)
        mindist = float(np.min(dmat))
        if verbosity > 0:
            print(
                f"{self.name}: Minimum distance between turbines: {mindist:.2f} m, minimum required: {self.stepsize_ortho:.2f} m"
            )

        # evaluate success criteria:
        mdist = min(
            self.stepsize_ortho,
            self.stepsize_wd if self.stepsize_wd is not None else np.inf,
        )
        success = (
            i == self._n_turbines
            and not np.any(np.isnan(pts))
            and mindist >= mdist - 1e-10
        )

        return success, pts
