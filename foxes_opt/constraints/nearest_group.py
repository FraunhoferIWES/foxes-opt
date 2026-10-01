from collections.abc import Sequence
from typing import Any

import foxes.constants as FC
import foxes.variables as FV
import numpy as np

from foxes_opt.core.farm_constraint import FarmConstraint
from foxes_opt.core.farm_opt_problem import FarmOptProblem


class NearestGroupConstraint(FarmConstraint):
    """
    Keep every turbine in a sufficiently large nearby connected group.

    For every selected turbine, this constraint finds the smallest connection
    radius at which its distance graph component reaches the required group
    size. The constraint value is that radius minus ``max_nearest_dist``.
    Requiring a group size of at least two also enforces the nearest-turbine
    upper bound when ``min_nearest_group_size`` is one.

    The connection radius is piecewise equal to the length of one graph edge.
    Its analytical derivative follows that active edge. At equal edge lengths,
    a deterministic active branch is used.
    """

    def __init__(
        self,
        problem: FarmOptProblem,
        max_nearest_dist: float,
        min_nearest_group_size: int,
        name: str = "nearest_group",
        sel_turbines: list[int] | None = None,
        infer_vars: bool = True,
        check_only_selected: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying optimization problem.
        max_nearest_dist
            Maximum graph-edge distance for connected turbine groups.
        min_nearest_group_size
            Minimum number of turbines in every connected group.
        name
            The name of the constraint.
        sel_turbines
            The selected turbines.
        infer_vars
            Whether to infer the float variables automatically.
        check_only_selected
            Whether groups may contain only selected turbines. By default,
            fixed turbines can contribute to a selected turbine's group.
        kwargs
            Additional parameters for ``iwopy.Constraint``.
        """
        if max_nearest_dist <= 0.0:
            raise ValueError(
                f"Constraint '{name}': max_nearest_dist must be positive, "
                f"got {max_nearest_dist}"
            )
        if min_nearest_group_size < 1:
            raise ValueError(
                f"Constraint '{name}': min_nearest_group_size must be at least 1, "
                f"got {min_nearest_group_size}"
            )

        self.max_nearest_dist = max_nearest_dist
        self.min_nearest_group_size = min_nearest_group_size
        self.infer_vars = infer_vars
        self.check_only_selected = check_only_selected
        self._required_group_size = max(2, min_nearest_group_size)
        self._checked_turbines = np.empty(0, dtype=int)
        self._component_checked_indices = np.empty(0, dtype=int)
        self._edge_indices = np.empty((0, 2), dtype=int)
        self._xy_var_indices = np.empty((0, 2), dtype=int)
        self._base_xy = np.empty((0, 2), dtype=np.float64)
        self._gradient_vars_float = np.empty(0, dtype=np.float64)
        self._gradient_xy = np.empty((0, 2), dtype=np.float64)
        self._gradient_edges = np.empty((0, 2), dtype=int)

        selected = problem.sel_turbines if sel_turbines is None else sel_turbines
        variable_names = [
            problem.tvar(coord, turbine)
            for turbine in selected
            for coord in (FV.X, FV.Y)
        ]
        super().__init__(
            problem,
            name,
            sel_turbines,
            vnames_float=None if infer_vars else variable_names,
            **kwargs,
        )

    def initialize(self, verbosity: int = 0) -> None:
        """
        Initialize the constraint.

        Parameters
        ----------
        verbosity
            The verbosity level, where zero is silent.
        """
        checked = (
            self.sel_turbines
            if self.check_only_selected
            else list(range(self.farm.n_turbines))
        )
        if self._required_group_size > len(checked):
            raise ValueError(
                f"Constraint '{self.name}': Required nearest group size "
                f"{self._required_group_size} exceeds the number of checked "
                f"turbines ({len(checked)})"
            )

        self._checked_turbines = np.asarray(checked, dtype=int)
        checked_indices = np.full(self.farm.n_turbines, -1, dtype=int)
        checked_indices[self._checked_turbines] = np.arange(len(checked), dtype=int)
        self._component_checked_indices = checked_indices[self.sel_turbines]
        self._edge_indices = np.column_stack(
            np.triu_indices(len(self._checked_turbines), k=1)
        )
        self._cnames = [f"{self.name}_{ti}" for ti in self.sel_turbines]
        self._base_xy = np.asarray(
            [
                np.asarray(turbine.xy).reshape(-1, 2)[0]
                for turbine in self.farm.turbines
            ],
            dtype=np.float64,
        )

        super().initialize(verbosity)
        self._xy_var_indices = np.full((self.farm.n_turbines, 2), -1, dtype=int)
        for variable_index, variable_name in enumerate(self.var_names_float):
            try:
                coordinate, turbine = self.problem.parse_tvar(variable_name)
            except (IndexError, ValueError):
                continue
            if coordinate in (FV.X, FV.Y):
                column = 0 if coordinate == FV.X else 1
                self._xy_var_indices[turbine, column] = variable_index
        self._gradient_vars_float = np.empty(0, dtype=np.float64)

    def n_components(self) -> int:
        """
        Return the number of constraint components.

        Returns
        -------
        value
            The number of selected turbines.
        """
        return self.n_sel_turbines

    def vardeps_float(self) -> np.ndarray:
        """
        Return float-variable dependencies for all components.

        Returns
        -------
        dependencies
            The dependency mask, with shape ``(n_components, n_vars_float)``.
        """
        dependencies = np.zeros((self.n_components(), self.n_vars_float), dtype=bool)
        checked = np.zeros(self.farm.n_turbines, dtype=bool)
        checked[self._checked_turbines] = True
        for variable, variable_name in enumerate(self.var_names_float):
            try:
                coordinate, turbine = self.problem.parse_tvar(variable_name)
            except (IndexError, ValueError):
                dependencies[:, variable] = True
                continue
            if checked[turbine]:
                dependencies[:, variable] = True
        return dependencies

    def component_turbines(
        self, components: Sequence[int] | np.ndarray | None = None
    ) -> set[int]:
        """
        Return turbines represented by constraint components.

        Parameters
        ----------
        components
            The selected component indices, or ``None`` for all components.

        Returns
        -------
        turbines
            The represented turbine indices.
        """
        component_indices = (
            np.arange(self.n_components(), dtype=int)
            if components is None
            else np.asarray(components, dtype=int)
        )
        selected = np.asarray(self.sel_turbines, dtype=int)
        return set(selected[component_indices].tolist())

    def _find_root(self, parents: np.ndarray, index: int) -> int:
        """Find a disjoint-set root and compress its path."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    def _connection_data(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate selected turbines' connection radii and active edges."""
        points = xy[self._checked_turbines]
        edge_deltas = (
            points[self._edge_indices[:, 0]] - points[self._edge_indices[:, 1]]
        )
        edge_distances = np.linalg.norm(edge_deltas, axis=1)
        edge_order = np.argsort(edge_distances, kind="stable")

        n_turbines = len(points)
        parents = np.arange(n_turbines, dtype=int)
        sizes = np.ones(n_turbines, dtype=int)
        unresolved = [[i] for i in range(n_turbines)]
        radii = np.full(n_turbines, np.nan, dtype=np.float64)
        active_edges = np.full((n_turbines, 2), -1, dtype=int)
        remaining = n_turbines

        for edge_index in edge_order:
            first, second = self._edge_indices[edge_index]
            first_root = self._find_root(parents, int(first))
            second_root = self._find_root(parents, int(second))
            if first_root == second_root:
                continue
            if sizes[first_root] < sizes[second_root]:
                first_root, second_root = second_root, first_root
            parents[second_root] = first_root
            sizes[first_root] += sizes[second_root]
            pending = unresolved[first_root] + unresolved[second_root]
            unresolved[second_root] = []
            if sizes[first_root] >= self._required_group_size:
                pending_indices = np.asarray(pending, dtype=int)
                radii[pending_indices] = edge_distances[edge_index]
                active_edges[pending_indices] = self._checked_turbines[[first, second]]
                unresolved[first_root] = []
                remaining -= len(pending)
                if remaining == 0:
                    break
            else:
                unresolved[first_root] = pending

        if np.any(np.isnan(radii)):
            raise RuntimeError(
                f"Constraint '{self.name}': Failed to determine connection radii"
            )
        indices = self._component_checked_indices
        return radii[indices], active_edges[indices]

    def _xy_from_vars(self, vars_float: np.ndarray) -> np.ndarray:
        """Apply function variables to a copy of the farm coordinates."""
        xy = self._base_xy.copy()
        for coordinate in range(2):
            variable_indices = self._xy_var_indices[:, coordinate]
            turbines = np.flatnonzero(variable_indices >= 0)
            xy[turbines, coordinate] = vars_float[variable_indices[turbines]]
        return xy

    def _get_gradient_data(
        self, vars_float: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate and cache coordinates and active connection edges."""
        if not np.array_equal(vars_float, self._gradient_vars_float):
            self._gradient_vars_float = vars_float.copy()
            self._gradient_xy = self._xy_from_vars(vars_float)
            _, self._gradient_edges = self._connection_data(self._gradient_xy)
        return self._gradient_xy, self._gradient_edges

    def ana_deriv(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        var: int,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate the analytical derivative for one float variable.

        Parameters
        ----------
        vars_int
            The integer variable values.
        vars_float
            The float variable values.
        var
            The float variable index.
        components
            The selected components, or ``None`` for all.

        Returns
        -------
        derivative
            The derivative values for the selected components.
        """
        component_indices = (
            np.arange(self.n_components(), dtype=int)
            if components is None
            else np.asarray(components, dtype=int)
        )
        derivative = np.zeros(len(component_indices), dtype=np.float64)
        try:
            coordinate_name, turbine = self.problem.parse_tvar(
                self.var_names_float[var]
            )
        except (IndexError, ValueError):
            return np.full(len(component_indices), np.nan, dtype=np.float64)
        if coordinate_name not in (FV.X, FV.Y):
            return np.full(len(component_indices), np.nan, dtype=np.float64)

        xy, active_edges = self._get_gradient_data(vars_float)
        edges = active_edges[component_indices]
        affected = np.any(edges == turbine, axis=1)
        if not np.any(affected):
            return derivative

        affected_edges = edges[affected]
        delta = xy[affected_edges[:, 0]] - xy[affected_edges[:, 1]]
        distance = np.linalg.norm(delta, axis=1)
        values = np.full(len(distance), np.nan, dtype=np.float64)
        nonzero = distance > 0.0
        coordinate = 0 if coordinate_name == FV.X else 1
        sign = np.where(affected_edges[:, 0] == turbine, 1.0, -1.0)
        values[nonzero] = sign[nonzero] * delta[nonzero, coordinate] / distance[nonzero]
        derivative[affected] = values
        return derivative

    def _result_xy(self, problem_results: Any) -> np.ndarray:
        """Extract state-independent coordinates from individual results."""
        xy = np.stack(
            [
                problem_results[FV.X].to_numpy(),
                problem_results[FV.Y].to_numpy(),
            ],
            axis=-1,
        )
        if not np.all(np.abs(np.min(xy, axis=0) - np.max(xy, axis=0)) < 1e-13):
            raise ValueError(f"Constraint '{self.name}': Require state independent XY")
        return xy[0]

    def calc_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: Any,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate constraint values for one individual.

        Parameters
        ----------
        vars_int
            The integer variable values.
        vars_float
            The float variable values.
        problem_results
            The results of applying the variables to the problem.
        components
            The selected components, or ``None`` for all.

        Returns
        -------
        values
            The selected constraint values.
        """
        xy = (
            self._result_xy(problem_results)
            if self.infer_vars
            else self._xy_from_vars(vars_float)
        )
        radii, _ = self._connection_data(xy)
        if components is not None:
            radii = radii[np.asarray(components, dtype=int)]
        return radii - self.max_nearest_dist

    def calc_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        problem_results: Any,
        components: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate constraint values for a population.

        Parameters
        ----------
        vars_int
            The integer variable values.
        vars_float
            The float variable values.
        problem_results
            The results of applying the population to the problem.
        components
            The selected components, or ``None`` for all.

        Returns
        -------
        values
            The selected constraint values for each individual.
        """
        n_pop = len(vars_float)
        if self.infer_vars:
            n_states = int(problem_results["n_org_states"].values)
            n_turbines = problem_results.sizes[FC.TURBINE]
            xy = np.stack(
                [
                    problem_results[FV.X].to_numpy(),
                    problem_results[FV.Y].to_numpy(),
                ],
                axis=-1,
            ).reshape(n_pop, n_states, n_turbines, 2)
            if not np.all(np.abs(np.min(xy, axis=1) - np.max(xy, axis=1)) < 1e-13):
                raise ValueError(
                    f"Constraint '{self.name}': Require state independent XY"
                )
            xy = xy[:, 0]
        else:
            xy = np.stack([self._xy_from_vars(values) for values in vars_float])

        component_indices = (
            np.arange(self.n_components(), dtype=int)
            if components is None
            else np.asarray(components, dtype=int)
        )
        values = np.empty((n_pop, len(component_indices)), dtype=np.float64)
        for population_index in range(n_pop):
            radii, _ = self._connection_data(xy[population_index])
            values[population_index] = radii[component_indices]
        return values - self.max_nearest_dist
