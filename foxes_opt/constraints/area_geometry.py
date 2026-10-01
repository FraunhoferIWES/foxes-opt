from collections.abc import Sequence
from typing import Any

import foxes.variables as FV
import numpy as np
from foxes.utils.geom2d import AreaGeometry

from foxes_opt.core.area_geometry_derivatives import signed_distance_derivatives
from foxes_opt.core.farm_constraint import FarmConstraint
from foxes_opt.core.farm_opt_problem import FarmOptProblem


class AreaGeometryConstraint(FarmConstraint):
    """
    Constrains turbine positions to the inside
    of a given area geometry.
    """

    def __init__(
        self,
        problem: FarmOptProblem,
        name: str,
        geometry: AreaGeometry,
        sel_turbines: list[int] | None = None,
        disc_inside: bool = False,
        D: float | None = None,
        infer_vars: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying optimization problem
        name
            The name of the constraint
        geometry
            The area geometry
        sel_turbines
            The selected turbines
        disc_inside
            Ensure full rotor disc inside boundary
        D
            Use this radius for rotor disc inside condition
        infer_vars
            Whether to infer the float variables automatically
        kwargs
            Additional parameters for `iwopy.Constraint`

        """
        self.geometry = geometry
        self.disc_inside = disc_inside
        self.infer_vars = infer_vars
        self.D = D

        selt = problem.sel_turbines if sel_turbines is None else sel_turbines
        vrs = []
        cns = []
        for ti in selt:
            vrs += [problem.tvar(FV.X, ti), problem.tvar(FV.Y, ti)]
            cns.append(f"{name}_{ti:04d}")
        vnames_float: list[str] | None = vrs
        if infer_vars:
            vnames_float = None

        super().__init__(
            problem,
            name,
            sel_turbines,
            vnames_float=vnames_float,
            cnames=cns,
            **kwargs,
        )

    def n_components(self) -> int:
        """
        Returns the number of components of the
        function.

        Returns
        -------
        value
            The number of components.

        """
        return self.n_sel_turbines

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
        deps = np.zeros((self.n_components(), self.n_components(), 2), dtype=bool)
        np.fill_diagonal(deps[:, :, 0], True)
        np.fill_diagonal(deps[:, :, 1], True)
        return deps.reshape(self.n_components(), self.n_components() * 2)

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
            The integer variable values
        vars_float
            The float variable values
        var
            The float variable index
        components
            The selected components, or None for all

        Returns
        -------
        deriv
            The derivative values, shape: (n_sel_components,)

        """
        cidx = (
            np.arange(self.n_components(), dtype=int)
            if components is None
            else np.asarray(components, dtype=int)
        )
        deriv = np.zeros(len(cidx), dtype=np.float64)
        try:
            vname, ti = self.problem.parse_tvar(self.var_names_float[var])
        except (IndexError, ValueError):
            return np.full(len(cidx), np.nan, dtype=np.float64)
        if vname not in (FV.X, FV.Y):
            return np.full(len(cidx), np.nan, dtype=np.float64)

        turbines = np.asarray(self.sel_turbines, dtype=int)[cidx]
        affected = turbines == ti
        if not np.any(affected):
            return deriv

        xy = np.asarray(
            [np.asarray(self.farm.turbines[t].xy).reshape(-1, 2)[0] for t in turbines],
            dtype=np.float64,
        )
        for name, value in zip(self.var_names_float, vars_float):
            try:
                xyname, xyti = self.problem.parse_tvar(name)
            except (IndexError, ValueError):
                continue
            target = np.flatnonzero(turbines == xyti)
            if xyname in (FV.X, FV.Y) and len(target):
                xy[target, 0 if xyname == FV.X else 1] = value

        coord = 0 if vname == FV.X else 1
        point_derivatives = np.zeros_like(xy[affected])
        point_derivatives[:, coord] = 1.0
        deriv[affected] = signed_distance_derivatives(
            self.geometry, xy[affected], point_derivatives
        )
        return deriv

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
        s: slice | np.ndarray[Any, np.dtype[np.int_]] = np.s_[:]
        if components is not None and len(components) < self.n_components():
            s = np.asarray(components, dtype=int)
        if self.infer_vars:
            xy = np.stack(
                [
                    problem_results[FV.X][:, self.sel_turbines][s],
                    problem_results[FV.Y][:, self.sel_turbines][s],
                ],
                axis=-1,
            )
            if not np.all(np.abs(np.min(xy, axis=0) - np.max(xy, axis=0)) < 1e-13):
                raise ValueError(
                    f"Constraint '{self.name}': Require state independet XY"
                )
            xy = xy[0]
        else:
            xy = vars_float.reshape(self.n_components(), 2)[s]

        dists = self.geometry.points_distance(xy)
        dists[self.geometry.points_inside(xy)] *= -1

        if self.disc_inside:
            if self.D is None:
                dists += problem_results[FV.D].to_numpy()[0, self.sel_turbines][s] / 2
            else:
                dists += self.D / 2

        return dists

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
        n_pop = len(vars_float)
        n_states = problem_results["n_org_states"].values
        n_cmpnts = self.n_components()
        s: slice | np.ndarray[Any, np.dtype[np.int_]] = np.s_[:]
        if components is not None and len(components) < self.n_components():
            n_cmpnts = len(components)
            s = np.asarray(components, dtype=int)
        if self.infer_vars:
            xy = np.stack(
                [
                    problem_results[FV.X][:, self.sel_turbines][:, s],
                    problem_results[FV.Y][:, self.sel_turbines][:, s],
                ],
                axis=-1,
            ).reshape(n_pop, n_states, n_cmpnts, 2)
            if not np.all(np.abs(np.min(xy, axis=1) - np.max(xy, axis=1)) < 1e-13):
                raise ValueError(
                    f"Constraint '{self.name}': Require state independet XY"
                )
            xy = xy[:, 0].reshape(n_pop * n_cmpnts, 2)
        else:
            xy = vars_float[:, s].reshape(n_pop * n_cmpnts, 2)

        dists = self.geometry.points_distance(xy)
        dists[self.geometry.points_inside(xy)] *= -1
        dists = dists.reshape(n_pop, n_cmpnts)

        if self.disc_inside:
            if self.D is None:
                D = problem_results[FV.D].to_numpy().reshape(n_pop, n_states, -1)
                D = D[:, :, self.sel_turbines][:, :, s]
                if not np.all(np.abs(np.min(D, axis=1) - np.max(D, axis=1)) < 1e-13):
                    raise ValueError(
                        f"Constraint '{self.name}': Require state independet D"
                    )
                dists += D[:, 0] / 2
            else:
                dists += self.D / 2

        return dists


class FarmBoundaryConstraint(AreaGeometryConstraint):
    """
    Constrains turbine positions to the inside of
    the wind farm boundary

    """

    def __init__(
        self, problem: FarmOptProblem, name: str = "boundary", **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying optimization problem
        name
            The name of the constraint
        kwargs
            Additional parameters for `AreaGeometryConstraint`

        """
        b = problem.farm.boundary
        assert b is not None, f"Constraint '{name}': Missing wind farm boundary."
        super().__init__(problem, name, geometry=b, **kwargs)
