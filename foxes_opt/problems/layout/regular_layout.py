import numpy as np
from copy import deepcopy

from foxes_opt.core import FarmVarsProblem, FarmOptProblem
from foxes.models.turbine_models import Calculator
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


def _calc_func(valid, P, ct, algo, mdata, fdata, st_sel):
    """helper function for Calculator turbine model"""
    return (valid, P * valid, ct * valid)


class RegularLayoutOptProblem(FarmVarsProblem):
    """
    Places turbines on a regular grid and optimizes
    its parameters.

    Attributes
    ----------
    min_spacing: float
        The minimal turbine spacing
    initial_values: dict
        Initial values for opt variables, key:
        spacing_x, spacing_y, offset_x, offset_y, angle

    :group: opt.problems.layout

    """

    SPACING_X = "spacing_x"
    SPACING_Y = "spacing_y"
    OFFSET_X = "offset_x"
    OFFSET_Y = "offset_y"
    ANGLE = "angle"

    def __init__(
        self,
        name,
        algo,
        min_spacing,
        initial_values=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        name: str
            The problem's name
        algo: foxes.core.Algorithm
            The algorithm
        min_spacing: float
            The minimal turbine spacing
        initial_values: dict, optional
            Initial values for opt variables, key:
            spacing_x, spacing_y, offset_x, offset_y, angle
        kwargs: dict, optional
            Additional parameters for `FarmVarsProblem`

        """
        super().__init__(name, algo, **kwargs)
        self.min_spacing = min_spacing
        self.initial_values = initial_values

    def initialize(self, verbosity=1, **kwargs):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Additional parameters for super class init

        """
        self._mname = self.name + "_calc"
        for t in self.algo.farm.turbines:
            if self._mname not in t.models:
                t.models.append(self._mname)
        self._turbine = deepcopy(self.farm.turbines[-1])

        self.algo.mbook.turbine_models[self._mname] = Calculator(
            in_vars=[FC.VALID, FV.P, FV.CT],
            out_vars=[FC.VALID, FV.P, FV.CT],
            func=_calc_func,
            pre_rotor=False,
        )

        b = self.farm.boundary
        assert b is not None, f"Problem '{self.name}': Missing wind farm boundary."
        pmax = b.p_max()
        pmin = b.p_min()
        self._pmin = pmin
        self._xy0 = pmin
        self._halfspan = (pmax - pmin) / 2
        self._halflen = np.linalg.norm(self._halfspan)
        self.max_spacing = 2 * (self._halflen + self.min_spacing)
        self._halfn = int(self._halflen / self.min_spacing)
        if self._halfn * self.min_spacing < self._halflen:
            self._halfn += 1
        self._nrow = 2 * self._halfn + 1
        self._nturb = self._nrow**2

        if verbosity > 0:
            print(f"Problem '{self.name}':")
            print(f"  xy0          = {self._xy0}")
            print(f"  span         = {np.linalg.norm(self._halfspan * 2):.2f}")
            print(f"  min spacing  = {self.min_spacing:.2f}")
            print(f"  max spacing  = {self.max_spacing:.2f}")
            print(f"  n row turbns = {self._nrow}")
            print(f"  n turbines   = {self._nturb}")
            print(f"  turbine mdls = {self._turbine.models}")

        iniv = self.initial_values
        self.initial_values = {
            self.SPACING_X: self.min_spacing,
            self.SPACING_Y: self.min_spacing,
            self.OFFSET_X: 0.0,
            self.OFFSET_Y: 0.0,
            self.ANGLE: 0.0,
        }
        if iniv is not None:
            mins = {
                v: m for v, m in zip(self.var_names_float(), self.min_values_float())
            }
            maxs = {
                v: m for v, m in zip(self.var_names_float(), self.max_values_float())
            }
            for k, v in iniv.items():
                assert k in self.initial_values, (
                    f"Invalid initial value key: '{k}', expected one of {list(self.initial_values.keys())}."
                )
                assert v >= mins[k], (
                    f"Initial value for '{k}' too small: {v} < {mins[k]}."
                )
                assert v <= maxs[k], (
                    f"Initial value for '{k}' too large: {v} > {maxs[k]}."
                )
                self.initial_values[k] = v
        if verbosity > 0:
            print("  initial values:")
            for k, v in self.initial_values.items():
                print(f"    {k:12s} = {v}")

        if self.farm.n_turbines < self._nturb:
            for i in range(self._nturb - self.farm.n_turbines):
                ti = len(self.farm.turbines)
                self.farm.turbines.append(deepcopy(self._turbine))
                self.farm.turbines[-1].index = ti
                self.farm.turbines[-1].name = f"T{ti}"
        elif self.farm.n_turbines > self._nturb:
            self.farm.turbines = self.farm.turbines[: self._nturb]
        self.algo.update_n_turbines()

        super().initialize(
            pre_rotor_vars=[FV.X, FV.Y, FC.VALID],
            post_rotor_vars=[],
            verbosity=verbosity,
            **kwargs,
        )

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names: list of str
            The names of the float variables

        """
        return [
            self.SPACING_X,
            self.SPACING_Y,
            self.OFFSET_X,
            self.OFFSET_Y,
            self.ANGLE,
        ]

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values: numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        return list(self.initial_values.values())

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        return np.array(
            [
                self.min_spacing,
                self.min_spacing,
                0.0,
                0.0,
                0.0,
            ],
            dtype=config.dtype_double,
        )

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        return np.array(
            [
                self.max_spacing,
                self.max_spacing,
                1.0,
                1.0,
                90.0,
            ],
            dtype=config.dtype_double,
        )

    def opt2farm_vars_individual(self, vars_int, vars_float):
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int: numpy.ndarray
            The integer optimization variable values,
            shape: (n_vars_int,)
        vars_float: numpy.ndarray
            The float optimization variable values,
            shape: (n_vars_float,)

        Returns
        -------
        farm_vars: dict
            The foxes farm variables. Key: var name,
            value: numpy.ndarray with values, shape:
            (n_states, n_sel_turbines)

        """

        dx, dy, ox, oy, a = vars_float
        n_states = self.algo.n_states
        nx = self._nrow
        ny = self._nrow

        a = np.deg2rad(a)
        nax = np.array([np.cos(a), np.sin(a), 0.0], dtype=config.dtype_double)
        nay = np.cross(np.array([0.0, 0.0, 1.0], dtype=config.dtype_double), nax)

        pts = np.zeros((nx, ny, 2), dtype=config.dtype_double)
        pts[:] = (
            self._xy0[None, None, :]
            + (ox + np.arange(nx)[:, None, None]) * dx * nax[None, None, :2]
            + (oy + np.arange(ny)[None, :, None]) * dy * nay[None, None, :2]
        )

        pts = pts.reshape(nx * ny, 2)
        valid = self.farm.boundary.points_inside(pts)

        farm_vars = {}
        for v, d in zip([FV.X, FV.Y, FC.VALID], [pts[:, 0], pts[:, 1], valid]):
            a = np.zeros((n_states, nx * ny), dtype=config.dtype_double)
            a[:] = d[None, :]
            farm_vars[v] = a

        return farm_vars

    def opt2farm_vars_population(self, vars_int, vars_float, n_states):
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int: numpy.ndarray
            The integer optimization variable values,
            shape: (n_pop, n_vars_int)
        vars_float: numpy.ndarray
            The float optimization variable values,
            shape: (n_pop, n_vars_float)
        n_states: int
            The number of original (non-pop) states

        Returns
        -------
        farm_vars: dict
            The foxes farm variables. Key: var name,
            value: numpy.ndarray with values, shape:
            (n_pop, n_states, n_sel_turbines)

        """
        n_pop = len(vars_float)
        n_turbines = self.farm.n_turbines
        dx = vars_float[:, 0]
        dy = vars_float[:, 1]
        ox = vars_float[:, 2]
        oy = vars_float[:, 3]
        nx = self._nrow
        ny = self._nrow
        a = vars_float[:, 4]
        N = self._nturb

        a = np.deg2rad(a)
        nax = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=-1)
        naz = np.zeros_like(nax)
        naz[..., 2] = 1
        nay = np.cross(naz, nax)

        pts = np.zeros((n_pop, nx, ny, 2), dtype=config.dtype_double)
        pts[:] = (
            self._xy0[None, None, None, :]
            + (ox[:, None, None, None] + np.arange(nx)[None, :, None, None])
            * dx[:, None, None, None]
            * nax[:, None, None, :2]
            + (oy[:, None, None, None] + np.arange(ny)[None, None, :, None])
            * dy[:, None, None, None]
            * nay[:, None, None, :2]
        )

        qts = np.zeros((n_pop, n_turbines, 2))
        qts[:, :N] = pts.reshape(n_pop, N, 2)
        del pts

        valid = self.farm.boundary.points_inside(
            qts.reshape(n_pop * n_turbines, 2)
        ).reshape(n_pop, n_turbines)

        farm_vars = {}
        for v, d in zip([FV.X, FV.Y, FC.VALID], [qts[:, :, 0], qts[:, :, 1], valid]):
            a = np.zeros((n_pop, n_states, n_turbines), dtype=config.dtype_double)
            a[:] = d[:, None, :]
            farm_vars[v] = a

        return farm_vars

    def finalize_individual(self, vars_int, vars_float, verbosity=1):
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int: np.array
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The optimal float variable values, shape: (n_vars_float,)
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem
        objs: np.array
            The objective function values, shape: (n_objectives,)
        cons: np.array
            The constraints values, shape: (n_constraints,)

        """
        farm_vars = self.opt2farm_vars_individual(vars_int, vars_float)
        sel = np.where(farm_vars[FC.VALID][0])[0]
        x = farm_vars[FV.X][0, sel]
        y = farm_vars[FV.Y][0, sel]

        turbines = [t for i, t in enumerate(self.farm.turbines) if i in sel]
        for i, t in enumerate(turbines):
            t.xy = np.array([x[i], y[i]], dtype=config.dtype_double)
            t.models = [m for m in t.models if m not in [self.name, self._mname]]
            t.index = i
            t.name = f"T{i}"
        self.farm.reset_turbines(self.algo, turbines)

        return FarmOptProblem.finalize_individual(
            self, vars_int, vars_float, verbosity=1
        )
