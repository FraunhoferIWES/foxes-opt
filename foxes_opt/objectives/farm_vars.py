from typing import Any

import numpy as np
import xarray as xr

from foxes_opt.core.farm_objective import FarmObjective
from foxes_opt.core.farm_opt_problem import FarmOptProblem
from foxes import variables as FV
import foxes.constants as FC


class FarmVarObjective(FarmObjective):
    """
    Objectives based on farm variables.

    Attributes
    ----------
    variable
        The variable name
    minimize
        Switch for maximizing or minimizing
    deps
        The foxes variables on which the variable depends,
        or None for all
    rules
        Contraction rules. Key: coordinate name str, value
        is: weights, mean_no_weights, sum, min, max
    scale
        The scaling factor


    """

    def __init__(
        self,
        problem: FarmOptProblem,
        name: str,
        variable: str,
        contract_states: str,
        contract_turbines: str,
        minimize: bool,
        deps: list[str] | None = None,
        scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        problem
            The underlying optimization problem
        name
            The name of the objective function
        variable
            The foxes variable name
        contract_states
            Contraction rule for states: min, max, sum, mean, weights
        contract_turbines
            Contraction rule for turbines: min, max, sum, mean
        minimize
            Switch for maximizing or minimizing
        deps
            The foxes variables on which the variable depends,
            or None for all
        scale
            The scaling factor
        kwargs
            Additional parameters for `FarmObjective`

        """
        super().__init__(problem, name, **kwargs)
        self.variable = variable
        self.minimize = minimize
        self.deps = deps
        self.scale = scale
        self.rules = {FC.STATE: contract_states, FC.TURBINE: contract_turbines}

    def initialize(self, verbosity: int = 0) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """
        super().initialize(verbosity)

    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        value
            The number of components.

        """
        return 1

    def maximize(self) -> list[bool]:
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags
            Bool array for component maximization,
            shape

        """
        return [not self.minimize]

    def vardeps_float(self) -> np.ndarray[tuple[int, int], np.dtype[np.bool_]]:
        """
        Gets the dependencies of all components
        on the function float variables

        Returns
        -------
        deps
            The dependencies of components on function
            variables, shape

        """
        if self.deps is None:
            return super().vardeps_float()

        out = np.zeros((self.n_components(), self.n_vars_float), dtype=bool)
        for i, tvr in enumerate(self.var_names_float):
            v, ti = self.problem.parse_tvar(tvr)
            if v in self.deps and ti in self.sel_turbines:
                out[0, i] = True

        return out

    def _contract(self, data: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
        """
        Helper function for data contraction
        """
        for dim, rule in self.rules.items():
            if rule == "min":
                data = data.min(dim=dim)
            elif rule == "max":
                data = data.max(dim=dim)
            elif rule == "sum":
                data = data.sum(dim=dim)
            elif rule == "mean_no_weights":
                data = data.mean(dim=dim)
            elif dim == FC.STATE and rule == "weights":
                odims = data.dims
                wdims = weights.dims
                if wdims == (FC.STATE,):
                    wx = "s"
                elif wdims == (FC.POP, FC.STATE):
                    wx = "ps"
                elif wdims == (FC.STATE, FC.TURBINE):
                    wx = "st"
                elif wdims == (FC.POP, FC.STATE, FC.TURBINE):
                    wx = "pst"
                else:
                    raise ValueError(
                        f"Objective '{self.name}': Expecting weight dimensions {(FC.STATE,)}, {(FC.POP, FC.STATE)}, {(FC.STATE, FC.TURBINE)} or {(FC.POP, FC.STATE, FC.TURBINE)}, got {wdims}"
                    )
                if len(odims) > 1 and odims[:2] == (FC.STATE, FC.TURBINE):
                    data = np.einsum(f"st...,{wx}->t...", data, weights)
                    data = xr.DataArray(data, dims=odims[1:])
                elif len(odims) > 2 and odims[:3] == (FC.POP, FC.STATE, FC.TURBINE):
                    data = np.einsum(f"pst...,{wx}->pt...", data, weights)
                    data = xr.DataArray(data, dims=(FC.POP,) + odims[2:])
                else:
                    raise NotImplementedError(
                        f"Contraction error for '{rule}' for dim '{dim}': Incompatible data dims {odims}, shape {data.shape}, for weights of shape {weights.shape}"
                    )
            elif dim == FC.STATE:
                raise ValueError(
                    f"Objective '{self.name}': Unknown contraction for dimension '{dim}': '{rule}'. Choose: weights, mean_no_weights, sum, min, max"
                )
            else:
                raise ValueError(
                    f"Objective '{self.name}': Unknown contraction for dimension '{dim}': '{rule}'. Choose: min, max, sum, mean_no_weights"
                )
        return data

    def calc_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: Any,
        components: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_sel_components,)

        """
        data = problem_results[self.variable]
        weights = problem_results[FV.WEIGHT]
        if self.n_sel_turbines < self.farm.n_turbines:
            data = data[:, self.sel_turbines]
        data = self._contract(data, weights) / self.scale

        return np.array([data], dtype=np.float64)

    def calc_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: Any,
        components: list[int] | None = None,
    ) -> np.ndarray:
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results
            The results of the variable application
            to the problem
        components
            The selected components or None for all

        Returns
        -------
        values
            The component values, shape: (n_pop, n_sel_components)

        """
        n_pop = problem_results["n_pop"].values
        n_states = problem_results["n_org_states"].values
        n_turbines = problem_results.sizes[FC.TURBINE]

        data = (
            problem_results[self.variable]
            .to_numpy()
            .reshape(n_states, n_pop, n_turbines)
        )
        data = np.swapaxes(data, 0, 1)
        data = xr.DataArray(data, dims=(FC.POP, FC.STATE, FC.TURBINE))

        weights = problem_results[FV.WEIGHT]
        wdims: tuple[str, ...]
        if weights.dims == (FC.STATE,):
            weights = problem_results[FV.WEIGHT].to_numpy().reshape(n_states, n_pop)
            weights = np.swapaxes(weights, 0, 1)
            wdims = (FC.POP, FC.STATE)
        elif weights.dims == (FC.STATE, FC.TURBINE):
            weights = (
                problem_results[FV.WEIGHT]
                .to_numpy()
                .reshape(n_states, n_pop, n_turbines)
            )
            weights = np.swapaxes(weights, 0, 1)
            wdims = (FC.POP, FC.STATE, FC.TURBINE)
        else:
            raise ValueError(
                f"Objective '{self.name}': Unsupported weight dimensions {weights.dims}"
            )
        weights = xr.DataArray(weights, dims=wdims)

        if self.n_sel_turbines < self.farm.n_turbines:
            data = data[:, self.sel_turbines]

        return self._contract(data / self.scale, weights).to_numpy()[:, None]

    def finalize_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: Any,
        verbosity: int = 1,
    ) -> np.ndarray:
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float
            The optimal float variable values, shape: (n_vars_float,)
        problem_results
            The results of the variable application
            to the problem
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        values
            The component values, shape: (n_components,)

        """
        return (
            super().finalize_individual(
                vars_int, vars_float, problem_results, verbosity
            )
            * self.scale
        )


class MaxFarmPower(FarmVarObjective):
    """
    Maximize the mean wind farm power

    Parameters
    ----------
    problem
        The underlying optimization problem
    name
        The name of the objective function
    kwargs
        Additional parameters for `FarmVarObjective`


    """

    def __init__(
        self, problem: FarmOptProblem, name: str = "maximize_power", **kwargs: Any
    ) -> None:
        if "scale" in kwargs:
            scale = kwargs.pop("scale")
        else:
            scale = 0.0
            ttypes = problem.algo.mbook.turbine_types
            for t in problem.farm.turbines:
                for mname in t.models:
                    if mname in ttypes:
                        scale += ttypes[mname].P_nominal
                        break

        super().__init__(
            problem,
            name,
            variable=FV.P,
            contract_states="weights",
            contract_turbines="sum",
            minimize=False,
            scale=scale,
            **kwargs,
        )


class MinimalMaxTI(FarmVarObjective):
    """
    Minimize the maximal turbine TI

    Parameters
    ----------
    problem
        The underlying optimization problem
    name
        The name of the objective function
    kwargs
        Additional parameters for `FarmVarObjective`


    """

    def __init__(
        self, problem: FarmOptProblem, name: str = "minimize_TI", **kwargs: Any
    ) -> None:
        scale = kwargs.pop("scale") if "scale" in kwargs else 1.0
        super().__init__(
            problem,
            name,
            variable=FV.TI,
            contract_states="max",
            contract_turbines="max",
            minimize=True,
            scale=scale,
            **kwargs,
        )
