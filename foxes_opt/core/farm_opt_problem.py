from typing import TYPE_CHECKING, Any

import numpy as np
from foxes.algorithms.downwind.models import PopulationStates
from foxes.config import config
from foxes.core import Algorithm, Engine, States, WindFarm, has_engine
from foxes.utils import new_instance
from iwopy import Problem

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class FarmOptProblem(Problem):
    """Abstract base class of FOXES-backed farm optimization problems."""

    def __init__(
        self,
        name: str,
        algo: Algorithm,
        sel_turbines: list[int] | None = None,
        calc_farm_args: dict[str, Any] | None = None,
        points: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a FOXES-backed optimization problem.

        Parameters
        ----------
        name
            Optimization problem name.
        algo
            FOXES algorithm retained and updated during candidate evaluations.
        sel_turbines
            Farm-order turbine indices to optimize, or ``None`` for all.
        calc_farm_args
            Additional parameters for ``algo.calc_farm()``. The default
            ``outputs="farm_results_all"`` preserves all calculated fields,
            including optimized farm variables.
        points
            Probe coordinates in metres with shape ``(n_states, n_points, 3)``,
            or ``None`` for farm-only evaluations.
        kwargs
            Additional parameters for ``iwopy.Problem``.

        """
        super().__init__(name, **kwargs)

        self.algo = algo
        self.calc_farm_args = {
            "outputs": "farm_results_all",
            **({} if calc_farm_args is None else calc_farm_args),
        }
        self.points = points

        self._sel_turbines = sel_turbines
        self._count: int | None = None

    @property
    def farm(self) -> WindFarm:
        """
        The wind farm

        Returns
        -------
        value
            The wind farm

        """
        return self.algo.farm

    @property
    def sel_turbines(self) -> list[int]:
        """
        The selected turbines

        Returns
        -------
        value
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
        value
            The numer of selected turbines

        """
        return len(self.sel_turbines)

    @property
    def all_turbines(self) -> bool:
        """
        Flag for all turbines optimization

        Returns
        -------
        value
            True if all turbines are subject to optimization

        """
        return len(self.sel_turbines) == self.algo.n_turbines

    @property
    def counter(self) -> int | None:
        """
        The current value of the application counter

        Returns
        -------
        value
            The current value of the application counter

        """
        return self._count

    @classmethod
    def tvar(cls, var: str, turbine_i: int) -> str:
        """
        Gets turbine variable name

        Parameters
        ----------
        var
            The variable name
        turbine_i
            The turbine index

        Returns
        -------
        value
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
        tvr
            The turbine variable name

        Returns
        -------
        var
            The foxes variable name
        turbine_i
            The turbine index

        """
        t = tvr.split("_")
        return t[0], int(t[1])

    def initialize(self, verbosity: int = 1) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
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
        """Replace states after finalizing and clearing loaded data and chunks."""
        if states is not self.algo.states:
            if self.algo.initialized:
                self.algo.finalize()
            self.algo.clear_loaded_data()
            self.algo.states = states
            self.algo.reset_chunk_store()

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
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
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
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int,)
        vars_float
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
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        problem_results
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
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        problem_results
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
            algo.verbosity = 0

            if self.points is None:
                point_results = None
            else:
                n_pop = len(vars_float)
                n_states, n_points = self.points.shape[:2]
                pop_points = np.zeros(
                    (n_pop, n_states, n_points, 3), dtype=config.dtype_double
                )
                pop_points[:] = self.points[None, :, :, :]
                pop_points = pop_points.reshape(n_states * n_pop, n_points, 3)
                point_results = algo.calc_points(farm_results, pop_points)

            farm_results["n_pop"] = len(vars_float)
            farm_results["n_org_states"] = self._org_n_states
            return (
                farm_results if point_results is None else (farm_results, point_results)
            )

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
        ax
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
        problem_type
            The selected derived class name
        args
            Additional parameters for the constructor
        kwargs
            Additional parameters for the constructor

        """
        return new_instance(cls, problem_type, *args, **kwargs)
