from typing import Any

from foxes.core import Algorithm, Output
from foxes.utils import Dict
from xarray import Dataset


class OptOutput(Output):
    """Base class for parent-side optimization-result outputs."""

    def __init__(
        self,
        to_file: str | None = None,
        write_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure optimization-result collection and optional NetCDF writing.

        Parameters
        ----------
        to_file
            NetCDF filename, or ``None`` to return the dataset without writing.
        write_pars
            Keyword arguments forwarded to ``write_nc``.
        kwargs
            Common FOXES output paths and model configuration.
        """
        super().__init__(fdata_variables=[], tdata_variables=[], **kwargs)
        self.to_file = to_file
        self.write_pars = {} if write_pars is None else dict(write_pars)

    @property
    def requires_chunk_calculation(self) -> bool:
        """Whether supplied results need worker calculations.

        Returns
        -------
        required
            Always false; iwopy results are processed on the parent.
        """
        return False

    def get_dataset(self) -> Dataset:
        """Translate optimization results into an xarray dataset.

        Returns
        -------
        dataset
            Optimization variables, objective and constraint components.
        """
        raise NotImplementedError

    def write_nc(self, fname: str, verbosity: int = 1, **kwargs: Any) -> None:
        """Write optimization results to NetCDF.

        Parameters
        ----------
        fname
            Filename resolved through the configured output directory.
        verbosity
            Verbosity level, where zero is silent.
        kwargs
            Options forwarded to ``foxes.utils.write_nc``.
        """
        raise NotImplementedError

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict,
    ) -> None:
        """Collect and optionally write optimization results on the parent.

        Parameters
        ----------
        algo
            Executing algorithm; no farm or point calculation is requested.
        n_chunks_states
            Number of state chunks, unused by optimization results.
        n_chunks_targets
            Number of target chunks, unused by optimization results.
        output_results
            Shared mapping receiving the dataset under this output's name.
        """
        output_results[self.name] = self.get_dataset()
        if self.to_file is not None:
            self.write_nc(self.to_file, **self.write_pars)
