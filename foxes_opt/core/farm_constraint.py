from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import foxes.variables as FV
import numpy as np
from foxes.core import WindFarm
from foxes.utils import all_subclasses, new_instance
from iwopy import Constraint

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .farm_opt_problem import FarmOptProblem


class FarmConstraint(Constraint):
    """
    Abstract base class for foxes wind farm
    optimization constraints.

    """

    def __init__(
        self,
        problem: "FarmOptProblem",
        name: str,
        sel_turbines: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            The underlying optimization problem
        name
            The name of the constraint
        sel_turbines
            The selected turbines
        kwargs
            Additional parameters for `iwopy.Constraint`

        """
        super().__init__(problem, name, **kwargs)
        self._sel_turbines = sel_turbines

    @property
    def farm(self) -> WindFarm:
        """
        The wind farm

        Returns
        -------
        value
            The wind farm

        """
        return self.problem.farm

    @property
    def sel_turbines(self) -> list[int]:
        """
        The list of selected turbines

        Returns
        -------
        value
            The list of selected turbines

        """
        return (
            self.problem.sel_turbines
            if self._sel_turbines is None
            else self._sel_turbines
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

    def component_turbines(
        self, components: Sequence[int] | np.ndarray | None = None
    ) -> set[int]:
        """
        Return turbines associated with constraint components.

        Parameters
        ----------
        components
            The selected component indices, or ``None`` for all components.

        Returns
        -------
        turbines
            The associated turbine indices.
        """
        component_indices = (
            np.arange(self.n_components(), dtype=int)
            if components is None
            else np.asarray(components, dtype=int)
        )
        dependencies = np.asarray(self.vardeps_float(), dtype=bool)[component_indices]
        variable_names = np.asarray(self.var_names_float)
        turbines: set[int] = set()
        for variable_name in variable_names[np.any(dependencies, axis=0)]:
            try:
                variable, turbine = self.problem.parse_tvar(variable_name)
            except (IndexError, ValueError):
                continue
            if variable in (FV.X, FV.Y):
                turbines.add(turbine)
        return turbines

    def add_to_layout_figure(self, ax: "Axes", **kwargs: Any) -> "Axes":
        """
        Add to a layout figure

        Parameters
        ----------
        ax
            The figure axis

        """
        return ax

    @classmethod
    def print_models(cls) -> None:
        """
        Prints all model names.
        """
        names = sorted([scls.__name__ for scls in all_subclasses(cls)])
        for n in names:
            print(n)

    @classmethod
    def new(cls, constraint_type: str, *args: Any, **kwargs: Any) -> Any:
        """
        Run-time farm constraint factory.

        Parameters
        ----------
        constraint_type
            The selected derived class name
        args
            Additional parameters for the constructor
        kwargs
            Additional parameters for the constructor

        """
        return new_instance(cls, constraint_type, *args, **kwargs)
