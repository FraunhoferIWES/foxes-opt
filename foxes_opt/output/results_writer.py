from xarray import Dataset
from iwopy.core import SingleObjOptResults, MultiObjOptResults
from foxes.utils import write_nc

from .opt_output import OptOutput


class SingleObjResultsWriter(OptOutput):
    """ 
    Writes optimization results to files.
    
    :group: output
    
    """

    def __init__(self, opt_results, **kwargs):
        """
        Constructor
        
        Parameters
        ----------
        opt_results: iwopy.core.SingleObjOptResults
            The optimization results
        kwargs: dict, optional
            Additional parameters for the base class
        
        """
        super().__init__(**kwargs)
        self.results = opt_results

        if not isinstance(opt_results, SingleObjOptResults):
            raise TypeError(f"{type(self).__name__}: Require results of type 'SingleObjOptResults', got '{type(opt_results).__name__}'")

    def get_dataset(self):
        """
        Translates results into dataset
        
        Returns
        -------
        ds: xarray.Dataset
            The results dataset

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
            attrs=dict(
                problem=self.results.pname,
                success=int(self.results.success),
                objective=self.results.onames[0],
                obj_value=self.results.objs,
            )
        )

        return ds

    def write_nc(self, fname, verbosity=1, **kwargs):
        """
        Writes to NetCDF file
        
        Parameters
        ----------
        fname: str
            Name of the file to be written
        verbosity: int
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Parameters for foxes.utils.write_nc
        
        """
        ds = self.get_dataset()
        fpath = self.get_fpath(fname)
        write_nc(ds, fpath, verbosity=verbosity, **kwargs)


class MultiObjResultsWriter(OptOutput):
    """ 
    Writes optimization results to files.
    
    :group: output
    
    """

    def __init__(self, opt_results, **kwargs):
        """
        Constructor
        
        Parameters
        ----------
        opt_results: iwopy.core.MultiObjOptResults
            The optimization results
        kwargs: dict, optional
            Additional parameters for the base class
        
        """
        super().__init__(**kwargs)
        self.results = opt_results

        if not isinstance(opt_results, MultiObjOptResults):
            raise TypeError(f"{type(self).__name__}: Require results of type 'MultiObjOptResults', got '{type(opt_results).__name__}'")

    def get_dataset(self):
        """
        Translates results into dataset
        
        Returns
        -------
        ds: xarray.Dataset
            The results dataset

        """
        
        crds = {}
        dvars = {}
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
            attrs=dict(
                problem=self.results.pname,
                success=int(self.results.success),
            )
        )

        return ds

    def write_nc(self, fname, verbosity=1, **kwargs):
        """
        Writes to NetCDF file
        
        Parameters
        ----------
        fname: str
            Name of the file to be written
        verbosity: int
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Parameters for foxes.utils.write_nc
        
        """
        ds = self.get_dataset()
        fpath = self.get_fpath(fname)
        write_nc(ds, fpath, verbosity=verbosity, **kwargs)
