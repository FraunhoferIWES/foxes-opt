foxes_opt.callbacks
===================
Contains optimizer-independent callbacks for wind farm optimization.

Continuing restart output
-------------------------

Use ``WriteLayoutCallback.step_offset`` to continue layout snapshot names after
a persisted index. Disable ``write_initial`` when the selected snapshot already
exists. Use ``WriteOptimizationHistoryCallback.iteration_offset`` with
``append=True`` to continue history rows. Append mode validates the existing
CSV header before preserving its rows; without append mode, initialization
replaces the history file.

.. code-block:: python

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
        ),
    ]

The first optimizer iteration then writes layout and history index 189.

.. toctree::
    :maxdepth: 2

    _autoapi/foxes_opt/callbacks/index
