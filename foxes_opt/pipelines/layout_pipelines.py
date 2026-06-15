from iwopy import Pipeline
from foxes import ModelBook


class LayoutPipeline(Pipeline):
    """
     Pipeline for layout generation.

     Attributes
     ----------
     algo_pars: dict
         The parameters for the foxes algorithm
     n_turbines: int
         The number of turbines in the wind farm
     turbine_models: list of str
         The turbine models
     mbook: foxes.ModelBook
         The model book
     farm_boundary: foxes.utils.geom2d.AreaGeometry
         The wind farm boundary

    :group: opt.pipelines

    """

    def __init__(
        self,
        base_dir,
        algo_pars,
        n_turbines,
        turbine_models,
        farm_boundary=None,
        mbook=None,
        name="layout_pipeline",
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        base_dir: str
            Base directory for the pipeline.
        algo_pars: dict
            The parameters for the foxes algorithm
        n_turbines: int
            The number of turbines in the wind farm
        turbine_models: list of str
            The turbine models
        farm_boundary: foxes.utils.geom2d.AreaGeometry, optional
            The wind farm boundary
        mbook: foxes.ModelBook, optional
            The model book to be used
        name: str, optional
            The name of the pipeline

        kwargs: dict
            Additional keyword arguments for the pipeline.

        """
        super().__init__(base_dir, name=name, **kwargs)
        self.algo_pars = algo_pars
        self.n_turbines = n_turbines
        self.farm_boundary = farm_boundary
        self.turbine_models = turbine_models
        self.mbook = mbook if mbook is not None else ModelBook()
