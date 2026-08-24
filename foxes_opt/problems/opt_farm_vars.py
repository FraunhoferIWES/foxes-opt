from typing import Any

import numpy as np
import pandas as pd

from foxes_opt.core import FarmVarsProblem
from foxes.models.turbine_models import SetFarmVars
from foxes.config import config


class OptFarmVars(FarmVarsProblem):
    """
    Optimize a selection of farm variables.


    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Constructor.

        Parameters
        ----------
        args
            Arguments for `FarmVarsProblem`
        kwargs
            Keyword arguments for `FarmVarsProblem`

        """
        super().__init__(*args, **kwargs)
        self._vars: pd.DataFrame | None = None

    def add_var(
        self,
        variable: str,
        typ: type[float] | type[int] | str,
        init: float | int,
        min: float | int,
        max: float | int,
        level: str = "uniform",
        sel: Any = None,
        model_key: str | None = None,
    ) -> None:
        """
        Add a variable.

        Parameters
        ----------
        variable
            The foxes farm variable name
        typ
            The variable type, either float or int
        init
            The initial value
        min
            The min value
        max
            The max value
        level
            Choices
        sel
            States/turbines/state-turbine selection,
            depending on the level
        model_key
            Creates sub-model which can then be placed in the
            turbine model list. Repeated keys are added to the
            same turbine model

        """
        if typ == "float":
            typ = float
        elif typ == "int":
            typ = int
        elif typ is not float and typ is not int:
            raise TypeError(
                f"Problem '{self.name}': Expecting float or int, got type '{type(typ).__name__}'"
            )

        mname = self.name if model_key is None else model_key
        if mname in self.algo.mbook.turbine_models:
            m = self.algo.mbook.turbine_models[mname]
            if not isinstance(m, SetFarmVars):
                raise KeyError(
                    f"Problem '{self.name}': Turbine model entry '{mname}' already exists in model book, and is not of type SetFarmVars"
                )
        else:
            self.algo.mbook.turbine_models[mname] = SetFarmVars()

        if self._vars is None:
            i0 = 0
            i0i = 0
            i0f = 0
        else:
            if variable in self._vars["var"].tolist():
                raise ValueError(
                    f"Problem '{self.name}': Attempt to add variable '{variable}' twice"
                )
            i0 = len(self._vars.index)
            grps = self._vars.groupby("type")
            i0i = len(grps.get_group("int").index) if "int" in grps.groups.keys() else 0
            i0f = (
                len(grps.get_group("float").index)
                if "float" in grps.groups.keys()
                else 0
            )
            del grps

        if level == "uniform":
            hdata = pd.DataFrame(index=[i0])
            hdata.loc[i0, "name"] = variable
            hdata.loc[i0, "var"] = variable
            hdata.loc[i0, "type"] = "int" if typ is int else "float"
            hdata.loc[i0, "index"] = i0i if typ is int else i0f
            hdata.loc[i0, "level"] = level
            hdata.loc[i0, "state"] = -1
            hdata.loc[i0, "turbine"] = -1
            hdata.loc[i0, "sel_turbine"] = -1
            hdata.loc[i0, "init"] = np.array([init], dtype=config.dtype_double)
            hdata.loc[i0, "min"] = np.array([min], dtype=config.dtype_double)
            hdata.loc[i0, "max"] = np.array([max], dtype=config.dtype_double)
            hdata.loc[i0, "model_key"] = mname

        elif level == "state":
            if not self.algo.initialized:
                self.algo.initialize()

            states = np.arange(self.algo.n_states)
            if sel is not None:
                states = states[sel]
            inds = i0 + np.arange(len(states))
            tinds = inds - i0 + i0i if typ is int else inds - i0 + i0f

            hdata = pd.DataFrame(index=inds)
            hdata.loc[inds, "name"] = [
                f"{variable}_{i:05d}" for i in range(len(states))
            ]
            hdata.loc[inds, "var"] = variable
            hdata.loc[inds, "type"] = "int" if typ is int else "float"
            hdata.loc[inds, "index"] = tinds
            hdata.loc[inds, "level"] = level
            hdata.loc[inds, "state"] = states
            hdata.loc[inds, "turbine"] = -1
            hdata.loc[inds, "sel_turbine"] = -1

            for c, d in [("init", init), ("min", min), ("max", max)]:
                data = np.full(len(inds), np.nan, dtype=config.dtype_double)
                data[:] = d
                hdata.loc[inds, c] = data

            hdata.loc[inds, "model_key"] = mname

        elif level == "turbine":
            if sel is None:
                turbines = self.sel_turbines
            else:
                turbines = np.arange(self.algo.n_turbines)[sel]
            inds = i0 + np.arange(len(turbines))
            tinds = inds - i0 + i0i if typ is int else inds - i0 + i0f

            hdata = pd.DataFrame(index=inds)
            hdata.loc[inds, "name"] = [
                f"{variable}_{i:04d}" for i in range(len(turbines))
            ]
            hdata.loc[inds, "var"] = variable
            hdata.loc[inds, "type"] = "int" if typ is int else "float"
            hdata.loc[inds, "index"] = tinds
            hdata.loc[inds, "level"] = level
            hdata.loc[inds, "state"] = -1
            hdata.loc[inds, "turbine"] = turbines
            hdata.loc[inds, "sel_turbine"] = [
                self.sel_turbines.index(ti) for ti in turbines
            ]

            for c, d in [("init", init), ("min", min), ("max", max)]:
                data = np.full(len(inds), np.nan, dtype=config.dtype_double)
                data[:] = d
                hdata.loc[inds, c] = data

            hdata.loc[inds, "model_key"] = mname

        elif level == "state-turbine":
            if not self.algo.initialized:
                self.algo.initialize()

            n_states = self.algo.n_states
            n_turbines = self.algo.n_turbines
            if sel is None:
                sel = np.zeros((n_states, n_turbines), dtype=bool)
                sel[:, self.sel_turbines] = True
            else:
                sel = np.array(sel, dtype=bool)
            st = np.arange(n_states * n_turbines).reshape(n_states, n_turbines)[sel]
            whr = np.where(sel)
            n_inds = len(st)
            inds = i0 + np.arange(n_inds)
            tinds = inds - i0 + i0i if typ is int else inds - i0 + i0f

            hdata = pd.DataFrame(index=inds)
            hdata.loc[inds, "name"] = [
                f"{variable}_{whr[0][i]:05d}_{whr[1][i]:04d}" for i in range(len(st))
            ]
            hdata.loc[inds, "var"] = variable
            hdata.loc[inds, "type"] = "int" if typ is int else "float"
            hdata.loc[inds, "index"] = tinds
            hdata.loc[inds, "level"] = level
            hdata.loc[inds, "state"] = whr[0]
            hdata.loc[inds, "turbine"] = whr[1]
            hdata.loc[inds, "sel_turbine"] = [
                self.sel_turbines.index(ti) for ti in whr[1]
            ]

            for c, d in [("init", init), ("min", min), ("max", max)]:
                data = np.full(n_inds, np.nan, dtype=config.dtype_double)
                if isinstance(d, np.ndarray) and len(d.shape) > 1:
                    data[:] = d[sel]
                else:
                    data[:] = d
                hdata.loc[inds, c] = data

            hdata.loc[inds, "model_key"] = mname

        else:
            raise ValueError(
                f"Problem '{self.name}': Unknown level '{level}'. Choices: uniform, state, turbine, state-turbine"
            )

        if self._vars is None:
            self._vars = hdata
        else:
            self._vars = pd.concat([self._vars, hdata], axis=0)

        icols = ["index", "state", "turbine", "sel_turbine"]
        for c in icols:
            self._vars[c] = self._vars[c].astype(config.dtype_int)

    def initialize(
        self,
        verbosity: int = 1,
        model_vars: dict[str, list[str]] | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Additional parameters for super class init

        """
        if self._vars is None:
            raise ValueError(
                f"Problem '{self.name}': No variables added for optimization."
            )

        if verbosity > 0:
            print(f"Problem '{self.name}': Optimization variable list")
            print()
            print(self._vars)
            print()

        vrs = {}
        for mname, g in self._vars.groupby("model_key"):
            if mname not in vrs:
                vrs[mname] = set(g["var"].tolist())
            else:
                vrs[mname].update(g["var"].tolist())

        super().initialize(
            model_vars={mname: list(vrs) for mname, vrs in vrs.items()},
            verbosity=verbosity,
            **kwargs,
        )

    def var_names_int(self) -> list[str]:
        """
        The names of int variables.

        Returns
        -------
        names
            The names of the int variables

        """
        if self._vars is None:
            raise ValueError(
                f"Problem '{self.name}': No variables added for optimization."
            )

        grps = self._vars.groupby("type")
        if "int" not in grps.groups.keys():
            return []
        else:
            return grps.get_group("int")["name"].tolist()

    def initial_values_int(self) -> np.ndarray:
        """
        The initial values of the int variables.

        Returns
        -------
        values
            Initial int values, shape: (n_vars_int,)

        """
        if self._vars is None:
            raise ValueError(
                f"Problem '{self.name}': No variables added for optimization."
            )

        grps = self._vars.groupby("type")
        if "int" not in grps.groups.keys():
            return np.array([], dtype=config.dtype_int)
        else:
            return grps.get_group("int")["init"].to_numpy(config.dtype_int)

    def min_values_int(self) -> np.ndarray:
        """
        The minimal values of the integer variables.

        Use -self.INT_INF for unbounded.

        Returns
        -------
        values
            Minimal int values, shape: (n_vars_int,)

        """
        if self._vars is None:
            raise ValueError(
                f"Problem '{self.name}': No variables added for optimization."
            )

        grps = self._vars.groupby("type")
        if "int" not in grps.groups.keys():
            return np.array([], dtype=config.dtype_int)
        else:
            return grps.get_group("int")["min"].to_numpy(config.dtype_int)

    def max_values_int(self) -> np.ndarray:
        """
        The maximal values of the integer variables.

        Use self.INT_INF for unbounded.

        Returns
        -------
        values
            Maximal int values, shape: (n_vars_int,)

        """
        if self._vars is None:
            raise ValueError(
                f"Problem '{self.name}': No variables added for optimization."
            )

        grps = self._vars.groupby("type")
        if "int" not in grps.groups.keys():
            return np.array([], dtype=config.dtype_int)
        else:
            return grps.get_group("int")["max"].to_numpy(config.dtype_int)

    def var_names_float(self) -> list[str]:
        """
        The names of float variables.

        Returns
        -------
        names
            The names of the float variables

        """
        if self._vars is None:
            raise ValueError(
                f"Problem '{self.name}': No variables added for optimization."
            )

        grps = self._vars.groupby("type")
        if "float" not in grps.groups.keys():
            return []
        else:
            return grps.get_group("float")["name"].tolist()

    def initial_values_float(self) -> np.ndarray:
        """
        The initial values of the float variables.

        Returns
        -------
        values
            Initial float values, shape: (n_vars_float,)

        """
        if self._vars is None:
            raise ValueError(
                f"Problem '{self.name}': No variables added for optimization."
            )

        grps = self._vars.groupby("type")
        if "float" not in grps.groups.keys():
            return np.array([], dtype=config.dtype_double)
        else:
            return grps.get_group("float")["init"].to_numpy(config.dtype_double)

    def min_values_float(self) -> np.ndarray:
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values
            Minimal float values, shape: (n_vars_float,)

        """
        if self._vars is None:
            raise ValueError(
                f"Problem '{self.name}': No variables added for optimization."
            )

        grps = self._vars.groupby("type")
        if "float" not in grps.groups.keys():
            return np.array([], dtype=config.dtype_double)
        else:
            return grps.get_group("float")["min"].to_numpy(config.dtype_double)

    def max_values_float(self) -> np.ndarray:
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values
            Maximal float values, shape: (n_vars_float,)

        """
        if self._vars is None:
            raise ValueError(
                f"Problem '{self.name}': No variables added for optimization."
            )

        grps = self._vars.groupby("type")
        if "float" not in grps.groups.keys():
            return np.array([], dtype=config.dtype_double)
        else:
            return grps.get_group("float")["max"].to_numpy(config.dtype_double)

    def opt2farm_vars_individual(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int
            The integer optimization variable values,
            shape
        vars_float
            The float optimization variable values,
            shape

        Returns
        -------
        farm_vars
            The foxes farm variables. Key: var name,
            value
            (n_states, n_sel_turbines)

        """
        if self._vars is None:
            raise ValueError(
                f"Problem '{self.name}': No variables added for optimization."
            )

        n_states = self.algo.n_states
        n_sturb = self.n_sel_turbines

        farm_vars = {}
        grps = self._vars.groupby(["type", "var", "level"])
        for (typ, var, level), g in grps:
            src = vars_int if typ == "int" else vars_float
            i0 = g.index[0]
            i1 = g.index[-1]
            data = src[np.s_[i0 : i1 + 1]]

            if level == "uniform":
                farm_vars[var] = np.full(
                    (n_states, n_sturb), data[0], dtype=config.dtype_double
                )

            elif level == "state":
                farm_vars[var] = np.full(
                    (n_states, n_sturb), np.nan, dtype=config.dtype_double
                )
                if np.all(g["state"] == np.arange(n_states)):
                    farm_vars[var][:] = data[:, None]
                else:
                    farm_vars[var][g["state"]] = data[:, None]

            elif level == "turbine":
                farm_vars[var] = np.full(
                    (n_states, n_sturb), np.nan, dtype=config.dtype_double
                )
                if np.all(g["sel_turbine"] == np.arange(n_sturb)):
                    farm_vars[var][:] = data[None, :]
                else:
                    farm_vars[var][:, g["sel_turbine"]] = data[None, :]

            elif level == "state-turbine":
                farm_vars[var] = np.full(
                    (n_states, n_sturb), np.nan, dtype=config.dtype_double
                )
                farm_vars[var][g["state"], g["sel_turbine"]] = data

            else:
                raise ValueError(
                    f"Problem '{self.name}': Unknown level '{level}' encountered for variable '{var}'. Valid choices: uniform, state, turbine, state-turbine"
                )

        return farm_vars

    def opt2farm_vars_population(
        self, vars_int: np.ndarray, vars_float: np.ndarray, n_states: int
    ) -> dict[str, np.ndarray]:
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int
            The integer optimization variable values,
            shape
        vars_float
            The float optimization variable values,
            shape
        n_states
            The number of original (non-pop) states

        Returns
        -------
        farm_vars
            The foxes farm variables. Key: var name,
            value
            (n_states, n_pop, n_sel_turbines)

        """
        if self._vars is None:
            raise ValueError(
                f"Problem '{self.name}': No variables added for optimization."
            )

        n_pop = vars_float.shape[0]
        n_sturb = self.n_sel_turbines

        farm_vars = {}
        grps = self._vars.groupby(["type", "var", "level"])
        for (typ, var, level), g in grps:
            src = vars_int if typ == "int" else vars_float
            i0 = g.index[0]
            i1 = g.index[-1]
            data = src[:, np.s_[i0 : i1 + 1]]

            if level == "uniform":
                farm_vars[var] = np.full(
                    (n_states, n_pop, n_sturb), np.nan, dtype=config.dtype_double
                )
                farm_vars[var][:] = data[None, :, 0, None]

            elif level == "state":
                farm_vars[var] = np.full(
                    (n_states, n_pop, n_sturb), np.nan, dtype=config.dtype_double
                )
                sdata = np.swapaxes(data[:, :, None], 0, 1)
                if np.all(g["state"] == np.arange(n_states)):
                    farm_vars[var][:] = sdata
                else:
                    farm_vars[var][g["state"]] = sdata

            elif level == "turbine":
                farm_vars[var] = np.full(
                    (n_states, n_pop, n_sturb), np.nan, dtype=config.dtype_double
                )
                if np.all(g["sel_turbine"] == np.arange(n_sturb)):
                    farm_vars[var][:] = data[None, :, :]
                else:
                    farm_vars[var][:, :, g["sel_turbine"]] = data[None, :, :]

            elif level == "state-turbine":
                farm_vars[var] = np.full(
                    (n_states, n_pop, n_sturb), np.nan, dtype=config.dtype_double
                )
                farm_vars[var][g["state"], :, g["sel_turbine"]] = data.T

            else:
                raise ValueError(
                    f"Problem '{self.name}': Unknown level '{level}' encountered for variable '{var}'. Valid choices: uniform, state, turbine, state-turbine"
                )

        return farm_vars
