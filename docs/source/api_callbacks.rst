foxes_opt.callbacks
===================
Contains backend-independent callbacks for wind farm optimization.

``WriteLayoutCallback`` requires a ``FarmOptProblem`` and accepts iteration or
evaluation events from callback-enabled iwopy optimizers. PyGMO IPOPT does not
support callbacks. Pipeline users pass an ordered callback list through
``LayoutOptimizerStage.callbacks``.

Layout snapshots
----------------

Pass ``--write_layouts`` to ``examples/layout_wind_rose/run_pymoo.py`` or
``run_slsqp.py`` to enable intermediate images; output is disabled by default.
These examples disable CSV output and write one image per generation or
accepted iteration under ``examples/layout_wind_rose/results/``, regardless of
the working directory. ``--layout_image_type`` selects the format (default:
``jpg``). Running an example again overwrites matching snapshot indices.

The callback selects the best first-objective individual. Each title names its
objective and value; constraint-valid turbines are blue and those associated
with violated constraints are red, with an upper-left legend. ``valid_color``
and ``invalid_color`` accept Matplotlib colors. Early snapshots may be
infeasible and are not final results.

``write_csv`` and ``write_image`` independently control output and both default
to ``True``. CSV files go directly into the output directory; images go into a
format-named child directory such as ``jpg/``. Images omit turbine annotations
and show physical turbine radii with contrasting outlines. The examples disable
per-file log messages, and generated output is excluded by ``.gitignore``.

The step-zero layout snapshot evaluates the optimizer's initial variables so
its objective title and constraint-validity colors are available before the
first optimizer iteration. Set ``write_initial=False`` to skip this evaluation
and snapshot.

Initial history rows
--------------------

``WriteOptimizationHistoryCallback`` records the selected objective value and
violated constraint-component count for each iteration, persisting each row
immediately. It resets the CSV when a solve starts unless append mode is enabled.

Set ``WriteOptimizationHistoryCallback(write_initial=True)`` to also record
the starting objective and violated constraint-component count in the history
CSV before the first optimizer iteration. This evaluates the optimizer's
initial variables during callback initialization. The history option defaults
to ``False`` and is independent of the initial layout snapshot.

.. code-block:: python

    from foxes_opt.callbacks import (
        WriteLayoutCallback,
        WriteOptimizationHistoryCallback,
    )

    callbacks = [
        WriteLayoutCallback("iterations", "layout", write_initial=True),
        WriteOptimizationHistoryCallback(
            "iterations/optimization_history.csv", write_initial=True
        ),
    ]

When these callbacks are passed to an optimizer, initialization writes layout
index zero and a history row with ``iteration=0``, ``objective``, and
``n_violated_constraints``. The normal iteration rows then begin at one.
The configured ``iteration_offset`` also applies to the initial history row.

Continuing restart output
-------------------------

Use ``WriteLayoutCallback.step_offset`` to continue layout snapshot names after
a persisted index. Disable ``write_initial`` when the selected snapshot already
exists. Use ``WriteOptimizationHistoryCallback.iteration_offset`` with
``append=True`` to continue history rows. Append mode validates the existing
CSV header before preserving its rows; without append mode, initialization
replaces the history file.
Leave the history callback's ``write_initial=False`` when the restart's
starting row is already present, avoiding a duplicate entry at that index.

.. code-block:: python

    from foxes_opt.callbacks import (
        WriteLayoutCallback,
        WriteOptimizationHistoryCallback,
    )

    callbacks = [
        WriteLayoutCallback(
            "iterations",
            "layout",
            write_initial=False,
            step_offset=188,
        ),
        WriteOptimizationHistoryCallback(
            "iterations/optimization_history.csv",
            iteration_offset=188,
            append=True,
            write_initial=False,
        ),
    ]

The first optimizer iteration then writes layout and history index 189.

.. toctree::
    :maxdepth: 2

    _autoapi/foxes_opt/callbacks/index
