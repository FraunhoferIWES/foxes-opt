API
===
Optimization problems define the basic wind farm problem setup and the variables that will be 
modified by the optimizer. Objectives and constraints are then subsequently added to the problem.
Finally, any optimizer from `iwopy <https://github.com/FraunhoferIWES/iwopy>`_ can be used to find 
optimal values of the optimization variables (or Pareto fronts). 

Please consult the `iwopy documentation<https://fraunhoferiwes.github.io/iwopy.docs/index.html>`_ for details 
of the optimizers, and see the :ref:`Examples` section for applications.

    .. table:: 
        :widths: auto

        =======================================  ============================================================
        Package                                  Description
        =======================================  ============================================================
        :ref:`foxes_opt.core`                    Abstract base classes and core functionality.
        :ref:`foxes_opt.problems`                Wind farm optimization problems.
        :ref:`foxes_opt.objectives`              Objectives for wind farm optimization problems.
        :ref:`foxes_opt.constraints`             Constraints for wind farm optimization problems.
        =======================================  ============================================================

foxes_opt.core
--------------
Contains core functionality and abstract base classes.

    .. python-apigen-group:: opt.core

foxes_opt.problems
------------------
Wind farm optimization problems.

    .. toctree::
        api_problems

foxes_opt.objectives
--------------------
Objectives for wind farm optimization problems.

    .. python-apigen-group:: opt.objectives

foxes_opt.constraints
---------------------
Constraints for wind farm optimization problems.

    .. python-apigen-group:: opt.constraints
