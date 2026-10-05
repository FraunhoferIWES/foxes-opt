foxes_opt.pipelines
===================
Pipelines for combining multiple wind farm optimizations.

Restarting a layout stage
-------------------------

``LayoutPipeline.run`` can seed the first selected stage from a persisted
layout callback snapshot. The numeric suffix is independent of zero padding:

.. code-block:: python

    success, results = pipeline.run(
        start_stage=1,
        restart_layout_index=188,
        restart_layout_dir="iterations",
    )

Relative restart directories resolve below the pipeline base directory. The
loader requires exactly one matching ``layout_<index>.csv`` with finite ``x``
and ``y`` coordinates, the expected turbine order when an ``index`` column is
present, and shape ``(n_turbines, 2)``.

Constraint-family parameters
----------------------------

``LayoutOptimizerStage`` keeps automatically generated minimum-distance
constraint parameters separate from explicit constraint entries. This permits
independent feasibility tolerances:

.. code-block:: python

    stage = LayoutOptimizerStage(
        optimizer_type="SLSQP",
        constraints=[
            {"constraint_type": "FarmBoundaryConstraint", "tol": 0.5},
        ],
        min_dist=2.5,
        min_dist_unit="D",
        min_dist_constraint_pars={"tol": 0.01},
    )

The constraint objects own these tolerances; the optimizer derives its solver
bounds from the registered constraints.

.. toctree::
    :maxdepth: 2

    _autoapi/foxes_opt/pipelines/index
