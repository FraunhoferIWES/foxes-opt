from typing import Any

from foxes.utils import write_nc
from iwopy.core import MultiObjOptResults, SingleObjOptResults
from numpy import ndarray
from xarray import Dataset

from .opt_output import OptOutput


class SingleObjResultsWriter(OptOutput):
    """Collect a single-objective solution as a dataset or NetCDF file."""

    def __init__(self, opt_results: SingleObjOptResults, **kwargs: Any) -> None:
        """Initialize the single-objective result writer.

        Parameters
        ----------
        opt_results
            iwopy single-objective results retained by the writer.
        kwargs
            Common output paths, ``to_file`` and ``write_pars``.

        Raises
        ------
        TypeError
            If ``opt_results`` is not a ``SingleObjOptResults`` instance.
        """
        super().__init__(**kwargs)
        self.results = opt_results

        if not isinstance(opt_results, SingleObjOptResults):
            raise TypeError(
                f"{type(self).__name__}: Require results of type 'SingleObjOptResults', got '{type(opt_results).__name__}'"
            )

    def get_dataset(self) -> Dataset:
        """Translate the single-objective solution into a dataset.

        Returns
        -------
        dataset
            Integer and float variables on ``variables_int`` and
            ``variables_float``, and constraint values on ``constraints``.
            Coordinates retain iwopy component order. The problem name,
            success flag, objective name and value are dataset attributes.
            Empty variable and constraint families are omitted.

        """

        crds = {}
        dvars = {}
        if len(self.results.vnames_int):
            crds["variables_int"] = self.results.vnames_int
            dvars["vars_int"] = ("variables_int", self.results.vars_int)
        if len(self.results.vnames_float):
            crds["variables_float"] = self.results.vnames_float
            dvars["vars_float"] = ("variables_float", self.results.vars_float)
        if len(self.results.cnames):
            crds["constraints"] = self.results.cnames
            dvars["cons"] = ("constraints", self.results.cons)

        ds = Dataset(
            coords=crds,
            data_vars=dvars,
            attrs={
                "problem": self.results.pname,
                "success": int(self.results.success),
                "objective": self.results.onames[0],
                "obj_value": self.results.objs,
            },
        )

        return ds

    def write_nc(self, fname: str, verbosity: int = 1, **kwargs: Any) -> None:
        """Write the single-objective solution to NetCDF.

        Parameters
        ----------
        fname
            Filename resolved through the configured output directory.
        verbosity
            Verbosity level, where zero is silent.
        kwargs
            Options forwarded to ``foxes.utils.write_nc``. An existing file
            at the resolved path is overwritten.

        """
        ds = self.get_dataset()
        fpath = self.get_fpath(fname)
        write_nc(ds, fpath, verbosity=verbosity, **kwargs)


class MultiObjResultsWriter(OptOutput):
    """Collect a multi-objective population as a dataset or NetCDF file."""

    def __init__(self, opt_results: MultiObjOptResults, **kwargs: Any) -> None:
        """Initialize the multi-objective result writer.

        Parameters
        ----------
        opt_results
            iwopy multi-objective results retained by the writer.
        kwargs
            Common output paths, ``to_file`` and ``write_pars``.

        Raises
        ------
        TypeError
            If ``opt_results`` is not a ``MultiObjOptResults`` instance.
        """
        super().__init__(**kwargs)
        self.results = opt_results

        if not isinstance(opt_results, MultiObjOptResults):
            raise TypeError(
                f"{type(self).__name__}: Require results of type 'MultiObjOptResults', got '{type(opt_results).__name__}'"
            )

    def get_dataset(self) -> Dataset:
        """Translate the multi-objective population into a dataset.

        Returns
        -------
        dataset
            Population-major variables, objectives and constraints on ``pop``
            and their named component dimensions, preserving iwopy order.
            ``success`` is an integer variable on ``pop`` retaining each
            candidate's success flag; the problem name is an attribute.
            Empty variable and component families are omitted.

        """

        crds = {}
        dvars: dict[str, tuple[str | tuple[str, ...], ndarray]] = {
            "success": ("pop", self.results.success.astype("int8"))
        }
        if len(self.results.vnames_int):
            crds["variables_int"] = self.results.vnames_int
            dvars["vars_int"] = (("pop", "variables_int"), self.results.vars_int)
        if len(self.results.vnames_float):
            crds["variables_float"] = self.results.vnames_float
            dvars["vars_float"] = (("pop", "variables_float"), self.results.vars_float)
        if len(self.results.onames):
            crds["objectives"] = self.results.onames
            dvars["objs"] = (("pop", "objectives"), self.results.objs)
        if len(self.results.cnames):
            crds["constraints"] = self.results.cnames
            dvars["cons"] = (("pop", "constraints"), self.results.cons)

        ds = Dataset(
            coords=crds,
            data_vars=dvars,
            attrs={
                "problem": self.results.pname,
            },
        )

        return ds

    def write_nc(self, fname: str, verbosity: int = 1, **kwargs: Any) -> None:
        """Write the multi-objective population to NetCDF.

        Parameters
        ----------
        fname
            Filename resolved through the configured output directory.
        verbosity
            Verbosity level, where zero is silent.
        kwargs
            Options forwarded to ``foxes.utils.write_nc``. An existing file
            at the resolved path is overwritten.

        """
        ds = self.get_dataset()
        fpath = self.get_fpath(fname)
        write_nc(ds, fpath, verbosity=verbosity, **kwargs)
