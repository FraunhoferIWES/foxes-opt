from abc import abstractmethod
from collections.abc import Sequence

import numpy as np


class ZeroFloatDerivatives:
    """Provide zero float derivatives for piecewise-constant functions."""

    @abstractmethod
    def n_components(self) -> int:
        """Return the number of function components."""

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
        n_components = self.n_components() if components is None else len(components)
        return np.zeros(n_components, dtype=np.float64)
