foxes_opt.callbacks
===================
Contains optimizer-independent callbacks for wind farm optimization.

The step-zero layout snapshot evaluates the optimizer's initial variables so
its objective title and constraint-validity colors are available before the
first optimizer iteration. Set ``write_initial=False`` to skip this evaluation
and snapshot.

Initial history rows
--------------------

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
