import numpy as np
from abc import abstractmethod
from typing import Any

from foxes.models.turbine_models import SetFarmVars
from foxes.config import config
from foxes.utils import new_instance
import foxes.variables as FV

from .farm_opt_problem import FarmOptProblem


class FarmVarsProblem(FarmOptProblem):
    """
    Abstract base class for models that optimize
    farm variables.

    :group: opt.core

    """

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
        model_vars: dict or list
            The variables to optimize. If dict, key: model name, value: list of variable names.
        verbosity: int
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Additional parameters for super class init

        """
        if model_vars is None:
            raise ValueError(
                f"Problem '{self.name}': Missing model_vars for initialization"
            )

        self._model_vars: dict[str, list[str]] = {}
        if isinstance(model_vars, dict):
            self._model_vars = {m: v for m, v in model_vars.items() if len(v)}
        elif len(model_vars):
            self._model_vars = {self.name: model_vars}

        cnt = 0
        for mname, vrs in self._model_vars.items():
            if mname in self.algo.mbook.turbine_models:
                m = self.algo.mbook.turbine_models[mname]
                if not isinstance(m, SetFarmVars):
                    raise KeyError(
                        f"FarmOptProblem '{self.name}': Turbine model entry '{mname}' already exists in model book, and is not of type SetFarmVars"
                    )
            else:
                self.algo.mbook.turbine_models[mname] = SetFarmVars()

            found = False
            for t in self.algo.farm.turbines:
                if mname in t.models:
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"FarmOptProblem '{self.name}': Missing entry '{mname}' among any of the turbine models"
                )
            cnt += len(vrs)
        if not cnt:
            raise ValueError(f"Problem '{self.name}': No variables to optimize")

        super().initialize(verbosity=verbosity, **kwargs)

    @abstractmethod
    def opt2farm_vars_individual(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> dict[str, np.ndarray]:
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
        pass

    @abstractmethod
    def opt2farm_vars_population(
        self, vars_int: np.ndarray, vars_float: np.ndarray, n_states: int
    ) -> dict[str, np.ndarray]:
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
            (n_states, n_pop, n_sel_turbines)

        """
        pass

    def update_problem_individual(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> None:
        """
        Update the algo and other data using
        the latest optimization variables.

        This function is called before running the farm
        calculation.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_vars_float,)

        """
        super().update_problem_individual(vars_int, vars_float)

        # prepare:
        n_states = self._org_n_states
        fvars = self.opt2farm_vars_individual(vars_int, vars_float)

        # update turbine model that sets vars to opt values:
        for mname, vrs in self._model_vars.items():
            model = self.algo.mbook.turbine_models[mname]
            model.reset()
            for v in vrs:
                vals = fvars.pop(v)
                if self.all_turbines:
                    model.add_var(v, vals)
                else:
                    data = np.zeros(
                        (n_states, self.algo.n_turbines), dtype=config.dtype_double
                    )
                    data[:, self.sel_turbines] = vals
                    model.add_var(v, data)

        if len(fvars):
            raise KeyError(
                f"Problem '{self.name}': Too many farm vars from opt2farm_vars_individual: {list(fvars.keys())}"
            )

    def update_problem_population(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> None:
        """
        Update the algo and other data using
        the latest optimization variables.

        This function is called before running the farm
        calculation.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float,)

        """
        super().update_problem_population(vars_int, vars_float)

        # prepare:
        n_pop = len(vars_float)
        n_states = self._org_n_states
        n_pstates = n_states * n_pop
        fvars = self.opt2farm_vars_population(vars_int, vars_float, n_states)

        # update turbine model that sets vars to opt values:
        for mname, vrs in self._model_vars.items():
            model = self.algo.mbook.turbine_models[mname]
            model.reset()
            for v in vrs:
                vals = fvars.pop(v)
                shp0 = list(vals.shape)
                shp1 = [n_pstates] + shp0[2:]
                if self.all_turbines:
                    model.add_var(v, vals.reshape(shp1))
                else:
                    data = np.zeros(
                        (n_pstates, self.algo.n_turbines), dtype=config.dtype_double
                    )
                    data[:, self.sel_turbines] = vals.reshape(shp1)
                    model.add_var(v, data)
                    del data

                # special case (x, y) needs to reshape turbine property. Value will be set by model
                if v in [FV.X, FV.Y]:
                    for ti in self.sel_turbines:
                        xy = self.algo.farm.turbines[ti].xy
                        if len(xy.shape) > 1 and xy.shape[0] != n_pstates:
                            self.algo.farm.turbines[ti].xy = np.full(
                                (n_pstates, 2), np.nan, dtype=config.dtype_double
                            )

        if len(fvars):
            raise KeyError(
                f"Problem '{self.name}': Too many farm vars from opt2farm_vars_population: {list(fvars.keys())}"
            )

    @classmethod
    def new(cls, problem_type: str, *args: Any, **kwargs: Any) -> Any:
        """
        Run-time farm vars opt problem factory.

        Parameters
        ----------
        problem_type: string
            The selected derived class name
        args: tuple, optional
            Additional parameters for the constructor
        kwargs: dict, optional
            Additional parameters for the constructor

        """
        return new_instance(cls, problem_type, *args, **kwargs)
