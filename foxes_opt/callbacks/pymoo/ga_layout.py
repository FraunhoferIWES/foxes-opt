from iwopy.interfaces.pymoo import DefaultCallbackTemplate
from foxes.output import FarmLayoutOutput
from foxes.config import get_output_path

from foxes_opt.core import FarmOptProblem

class GAWriteLayoutCallbackTemplate(DefaultCallbackTemplate):
    """
    Template for the layout writing callback for genetic algorithms.

    Parameters
    ----------
    out_dir : str
        The output directory where the layout will be written
    base_name : str
        The base name for the layout files
    n_gen_step : int
        The number of generations between writing the layout
    figsize : tuple, optional
        The figure size for the layout plot
    verbosity : int
        The verbosity level, 0 = silent

    :group: opt.callbacks.pymoo

    """

    CLASS_NAME = "GAWriteLayoutCallback"
    CLASS_DOC = "The layout writing callback for genetic algorithms"

    def __init__(
            self,
            out_dir, 
            base_name, 
            n_gen_step=1, 
            figsize=None, 
            from_farm_results=False, 
            verbosity=0,
        ):
        """
        Initialize the callback.

        Parameters
        ----------
        out_dir : str
            The output directory where the layout will be written
        base_name : str
            The base name for the layout files
        n_gen_step : int
            The number of generations between writing the layout
        figsize : tuple, optional
            The figure size for the layout plot
        verbosity : int
            The verbosity level, 0 = silent
        from_farm_results : bool
            Whether to get the layout from farm results

        """
        super().__init__()
        self.out_dir = out_dir
        self.base_name = base_name
        self.n_gen_step = n_gen_step
        self.figsize = figsize
        self.verbosity = verbosity
        self.from_farm_results = from_farm_results

        def notify(self, algorithm):
            super().notify(algorithm)
            if algorithm.n_gen % self.n_gen_step == 0:

                problem = algorithm.problem.problem
                assert isinstance(problem, FarmOptProblem), f"Expected problem of type {FarmOptProblem.__name__}, got {type(problem)}"
                algo = problem.algo
                farm = algo.farm

                bname = f"{self.base_name}_{algorithm.n_gen:05d}"

                if self.from_farm_results:
                    results = problem.current_problem_results
                    raise NotImplementedError(f"Getting the layout from farm results is not implemented yet {results}")
                    o = FarmLayoutOutput(farm, farm_results=results, from_results=True, results_state=0)
                else:
                    o = FarmLayoutOutput(farm)

                fpath = out_dir / f"{bname}.csv"
                if self.verbosity > 0:
                    print(f"Writing layout to {fpath}")
                o.write_csv(get_output_path(fpath), type_col="turbine_type", algo=algo)

                fpath = out_dir / f"{bname}.png"
                if self.verbosity > 0:
                    print(f"Writing layout to {fpath}")
                o.write_plot(get_output_path(fpath), figsize=self.figsize)