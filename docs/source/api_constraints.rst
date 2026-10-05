foxes_opt.constraints
=====================
Constraints for wind farm optimization problems.

Nearest connected groups
------------------------

``NearestGroupConstraint`` combines an upper nearest-turbine distance with a
minimum connected-group size. For each selected turbine, it constrains the
smallest distance threshold at which the turbine's graph component contains
``max(2, min_nearest_group_size)`` turbines. This permits chains of nearby
turbines while rejecting isolated turbines and undersized components.

The connection radius is piecewise equal to one active edge length, so the
constraint provides analytical derivatives for direct turbine X/Y variables.
At equal edge lengths, the deterministic active branch supplies the derivative.
Other layout parameterizations remain marked as dependent and can be handled by
``LocalFD``.

Configure it through ``LayoutOptimizerStage.constraints``. A custom constraint
list replaces the default boundary entry, so include that entry explicitly:

.. code-block:: python

    constraints = [
        {"constraint_type": "FarmBoundaryConstraint"},
        {
            "constraint_type": "NearestGroupConstraint",
            "max_nearest_dist": 1200.0,
            "min_nearest_group_size": 10,
        },
    ]

Constraint families own their tolerances independently. For example, combine a
boundary tolerance with separate automatic minimum-distance parameters:

.. code-block:: python

    stage = LayoutOptimizerStage(
        optimizer_type="SLSQP",
        constraints=[
            {"constraint_type": "FarmBoundaryConstraint", "tol": 0.5},
        ],
        min_dist_constraint_pars={"tol": 0.01},
    )

.. toctree::
    :maxdepth: 2

    _autoapi/foxes_opt/constraints/index
