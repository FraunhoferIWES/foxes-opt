from typing import Any, TYPE_CHECKING

import numpy as np
from iwopy import Problem

from foxes.algorithms.downwind.models import PopulationStates
from foxes.core import has_engine, Engine, Algorithm, States, WindFarm
from foxes.config import config
from foxes.utils import new_instance

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class FarmOptProblem(Problem):
    """
    Abstract base class of wind farm optimization problems.

    Attributes
    ----------
    algo: foxes.core.Algorithm
        The algorithm
    calc_farm_args: dict
        Additional parameters for algo.calc_farm()
    points : numpy.ndarray
        The probe points, shape: (n_states, n_points, 3)

    :group: opt.core

    """

    def __init__(
        self,
        name: str,
        algo: Algorithm,
        sel_turbines: list[int] | None = None,
        calc_farm_args: dict[str, Any] | None = None,
        points: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        name: str
            The problem's name
        algo: foxes.core.Algorithm
            The algorithm
        sel_turbines: list of int, optional
            The turbines selected for optimization,
            or None for all
        calc_farm_args: dict
            Additional parameters for algo.calc_farm()
        points: numpy.ndarray, optional
            The probe points, shape: (n_states, n_points, 3)
        kwargs: dict, optional
            Additional parameters for `iwopy.Problem`

        """
        super().__init__(name, **kwargs)

        self.algo = algo
        self.calc_farm_args = {} if calc_farm_args is None else calc_farm_args
        self.points = points

        self._sel_turbines = sel_turbines
        self._count: int | None = None

    @property
    def farm(self) -> WindFarm:
        """
        The wind farm

        Returns
        -------
        foxes.core.WindFarm :
            The wind farm

        """
        return self.algo.farm

    @property
    def sel_turbines(self) -> list[int]:
        """
        The selected turbines

        Returns
        -------
        list of int :
            Indices of the selected turbines

        """
        return (
            self._sel_turbines
            if self._sel_turbines is not None
            else list(range(self.farm.n_turbines))
        )

    @property
    def n_sel_turbines(self) -> int:
        """
        The numer of selected turbines

        Returns
        -------
        int :
            The numer of selected turbines

        """
        return len(self.sel_turbines)

    @property
    def all_turbines(self) -> bool:
        """
        Flag for all turbines optimization

        Returns
        -------
        bool :
            True if all turbines are subject to optimization

        """
        return len(self.sel_turbines) == self.algo.n_turbines

    @property
    def counter(self) -> int | None:
        """
        The current value of the application counter

        Returns
        -------
        int :
            The current value of the application counter

        """
        return self._count

    @classmethod
    def tvar(cls, var: str, turbine_i: int) -> str:
        """
        Gets turbine variable name

        Parameters
        ----------
        var: str
            The variable name
        turbine_i: int
            The turbine index

        Returns
        -------
        str :
            The turbine variable name

        """
        return f"{var}_{turbine_i:04d}"

    @classmethod
    def parse_tvar(cls, tvr: str) -> tuple[str, int]:
        """
        Parse foxes variable name and turbine index
        from turbine variable

        Parameters
        ----------
        tvr: str
            The turbine variable name

        Returns
        -------
        var: str
            The foxes variable name
        turbine_i: int
            The turbine index

        """
        t = tvr.split("_")
        return t[0], int(t[1])

    def initialize(self, verbosity: int = 1) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        if not self.algo.initialized:
            self.algo.initialize()
        self._org_states_name = self.algo.states.name
        self._org_n_states = self.algo.n_states

        self.algo.finalize()

        self._count = 0

        super().initialize(verbosity)

    def _reset_states(self, states: States) -> None:
        """
        Reset the states in the algorithm
        """
        if states is not self.algo.states:
            if hasattr(self.algo, "clear_loaded_data"):
                self.algo.clear_loaded_data()
            if self.algo.initialized:
                self.algo.finalize()
            self.algo.states = states
            if hasattr(self.algo, "reset_chunk_store"):
                self.algo.reset_chunk_store()

            # Compatibility shim for older foxes versions that keep model data
            # in idata_mem/get_model_data but do not expose loaded_data.
            if not hasattr(self.algo, "loaded_data") and hasattr(
                self.algo, "get_model_data"
            ):
                try:
                    idata = self.algo.get_model_data(self.algo.states)
                except Exception:
                    idata = {"coords": {}, "data_vars": {}}
                data_vars = dict(idata.get("data_vars", {}))
                if (
                    "PopulationStates_SMAP" in data_vars
                    and "PopulationStates_smap" not in data_vars
                ):
                    data_vars["PopulationStates_smap"] = data_vars[
                        "PopulationStates_SMAP"
                    ]
                setattr(
                    self.algo,
                    "loaded_data",
                    {
                        "coords": dict(idata.get("coords", {})),
                        "data_vars": data_vars,
                        "extra_data": {},
                    },
                )

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
        # reset states, if needed:
        if isinstance(self.algo.states, PopulationStates):
            self._reset_states(self.algo.states.states)
            self.algo.n_states = self._org_n_states

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
        # Always rebuild the population-states wrapper. Repeated vectorized
        # evaluations (e.g. solver finalization) must start from a clean
        # mapping between original and expanded state dimensions.
        n_pop: int = len(vars_float)
        if not isinstance(self.algo.states, PopulationStates):
            ostates = self.algo.states
        else:
            ostates = self.algo.states.states
        self._reset_states(PopulationStates(ostates, n_pop))

    def apply_individual(self, vars_int: np.ndarray, vars_float: np.ndarray) -> Any:
        """
        Apply new variables to the problem.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem

        """
        if self._count is None:
            raise RuntimeError(f"Problem '{self.name}' not initialized")
        self._count += 1
        self.update_problem_individual(vars_int, vars_float)

        def _run_calc(algo: Algorithm) -> Any:
            """Helper function to run main foxes calculations"""
            farm_results = algo.calc_farm(**self.calc_farm_args)
            algo.verbosity = 0
            if self.points is None:
                return farm_results
            else:
                point_results = algo.calc_points(farm_results, self.points)
                return farm_results, point_results

        if has_engine():
            results = _run_calc(self.algo)
        else:
            with Engine.new("default", verbosity=0):
                results = _run_calc(self.algo)

        return results

    def apply_population(self, vars_int: np.ndarray, vars_float: np.ndarray) -> Any:
        """
        Apply new variables to the problem,
        for a whole population.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem

        """
        if self._count is None:
            raise RuntimeError(f"Problem '{self.name}' not initialized")
        self._count += 1

        self.update_problem_population(vars_int, vars_float)

        def _run_calc(algo: Algorithm) -> Any:
            """Helper function to run main foxes calculations"""
            farm_results = algo.calc_farm(**self.calc_farm_args)
            farm_results["n_pop"] = len(vars_float)
            farm_results["n_org_states"] = self._org_n_states
            algo.verbosity = 0

            if self.points is None:
                return farm_results
            else:
                n_pop = farm_results["n_pop"].values
                n_states, n_points = self.points.shape[:2]
                pop_points = np.zeros(
                    (n_states, n_pop, n_points, 3), dtype=config.dtype_double
                )
                pop_points[:] = self.points[:, None, :, :]
                pop_points = pop_points.reshape(n_states * n_pop, n_points, 3)
                point_results = algo.calc_points(farm_results, pop_points)
                return farm_results, point_results

        if has_engine():
            results = _run_calc(self.algo)
        else:
            with Engine.new("default", verbosity=0):
                results = _run_calc(self.algo)

        return results

    def add_to_layout_figure(self, ax: "Axes", **kwargs: Any) -> "Axes":
        """
        Add to a layout figure

        Parameters
        ----------
        ax: matplotlib.pyplot.Axis
            The figure axis

        """
        for c in self.cons.functions:
            ax = c.add_to_layout_figure(ax, **kwargs)
        for f in self.objs.functions:
            ax = f.add_to_layout_figure(ax, **kwargs)

        return ax

    @classmethod
    def new(cls, problem_type: str, *args: Any, **kwargs: Any) -> Any:
        """
        Run-time farm opt problem factory.

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
