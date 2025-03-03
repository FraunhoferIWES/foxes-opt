import numpy as np
import xarray as xr

from foxes_opt.core.farm_objective import FarmObjective
from foxes import variables as FV
import foxes.constants as FC


class FarmVarObjective(FarmObjective):
    """
    Objectives based on farm variables.

    Attributes
    ----------
    variable: str
        The variable name
    minimize: bool
        Switch for maximizing or minimizing
    deps: list of str
        The foxes variables on which the variable depends,
        or None for all
    rules: dict
        Contraction rules. Key: coordinate name str, value
        is str: weights, mean_no_weights, sum, min, max
    scale: float
        The scaling factor

    :group: opt.objectives

    """

    def __init__(
        self,
        problem,
        name,
        variable,
        contract_states,
        contract_turbines,
        minimize,
        deps=None,
        scale=1.0,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes_opt.FarmOptProblem
            The underlying optimization problem
        name: str
            The name of the objective function
        variable: str
            The foxes variable name
        contract_states: str
            Contraction rule for states: min, max, sum, mean, weights
        contract_turbines: str
            Contraction rule for turbines: min, max, sum, mean
        minimize: bool
            Switch for maximizing or minimizing
        deps: list of str
            The foxes variables on which the variable depends,
            or None for all
        scale: float
            The scaling factor
        kwargs: dict, optional
            Additional parameters for `FarmObjective`

        """
        super().__init__(problem, name, **kwargs)
        self.variable = variable
        self.minimize = minimize
        self.deps = deps
        self.scale = scale
        self.rules = {FC.STATE: contract_states, FC.TURBINE: contract_turbines}

    def initialize(self, verbosity=0):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().initialize(verbosity)

    def n_components(self):
        """
        Returns the number of components of the
        function.

        Returns
        -------
        int:
            The number of components.

        """
        return 1

    def maximize(self):
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags: np.array
            Bool array for component maximization,
            shape: (n_components,)

        """
        return [not self.minimize]

    def vardeps_float(self):
        """
        Gets the dependencies of all components
        on the function float variables

        Returns
        -------
        deps: numpy.ndarray of bool
            The dependencies of components on function
            variables, shape: (n_components, n_vars_float)

        """
        if self.deps is None:
            return super().vardeps_float()

        out = np.zeros((self.n_components(), self.n_vars_float), dtype=bool)
        for i, tvr in enumerate(self.var_names_float):
            v, ti = self.problem.parse_tvar(tvr)
            if v in self.deps and ti in self.sel_turbines:
                out[0, i] = True

        return out

    def _contract(self, data, weights):
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

    def calc_individual(self, vars_int, vars_float, problem_results, components=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_vars_float,)
        problem_results: Any
            The results of the variable application
            to the problem
        components: list of int, optional
            The selected components or None for all

        Returns
        -------
        values: np.array
            The component values, shape: (n_sel_components,)

        """
        data = problem_results[self.variable]
        weights = problem_results[FV.WEIGHT]
        if self.n_sel_turbines < self.farm.n_turbines:
            data = data[:, self.sel_turbines]
        data = self._contract(data, weights) / self.scale

        return np.array([data], dtype=np.float64)

    def calc_population(self, vars_int, vars_float, problem_results, components=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results: Any
            The results of the variable application
            to the problem
        components: list of int, optional
            The selected components or None for all

        Returns
        -------
        values: np.array
            The component values, shape: (n_pop, n_sel_components)

        """
        n_pop = problem_results["n_pop"].values
        n_states = problem_results["n_org_states"].values
        n_turbines = problem_results.sizes[FC.TURBINE]

        data = (
            problem_results[self.variable]
            .to_numpy()
            .reshape(n_pop, n_states, n_turbines)
        )
        data = xr.DataArray(data, dims=(FC.POP, FC.STATE, FC.TURBINE))

        weights = problem_results[FV.WEIGHT]
        if weights.dims == (FC.STATE,):
            weights = problem_results[FV.WEIGHT].to_numpy().reshape(n_pop, n_states)
            wdims = (FC.POP, FC.STATE)
        elif weights.dims == (FC.STATE, FC.TURBINE):
            weights = (
                problem_results[FV.WEIGHT]
                .to_numpy()
                .reshape(n_pop, n_states, n_turbines)
            )
            wdims = (FC.POP, FC.STATE, FC.TURBINE)
        weights = xr.DataArray(weights, dims=wdims)

        if self.n_sel_turbines < self.farm.n_turbines:
            data = data[:, self.sel_turbines]

        return self._contract(data / self.scale, weights).to_numpy()[:, None]

    def finalize_individual(self, vars_int, vars_float, problem_results, verbosity=1):
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int: np.array
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The optimal float variable values, shape: (n_vars_float,)
        problem_results: Any
            The results of the variable application
            to the problem
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        values: np.array
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
    problem: foxes_opt.FarmOptProblem
        The underlying optimization problem
    name: str
        The name of the objective function
    kwargs: dict, optional
        Additional parameters for `FarmVarObjective`

    :group: opt.objectives

    """

    def __init__(self, problem, name="maximize_power", **kwargs):
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
    problem: foxes_opt.FarmOptProblem
        The underlying optimization problem
    name: str
        The name of the objective function
    kwargs: dict, optional
        Additional parameters for `FarmVarObjective`

    :group: opt.objectives

    """

    def __init__(self, problem, name="minimize_TI", **kwargs):
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
