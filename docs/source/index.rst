Welcome to FOXES Optimization
=============================

.. versionadded:: 0.9.1
    Restart selected layout stages from persisted snapshots while continuing
    layout and optimization-history output numbering.

.. versionchanged:: 0.9.1
    Compatibility with *foxes* version 1.9.8 and *iwopy* version 1.0.0.

Overview
--------

*FOXES Optimization* is a modular Python package for wind-farm optimization by
Fraunhofer IWES. Typical applications include:

* Wind-farm layout optimization,
* Wake steering through turbine yaw optimization,
* Constrained optimization of turbine positions and farm geometries,
* Multi-stage layout optimization pipelines.

FOXES Optimization combines wind-farm and wake modelling from
`FOXES <https://github.com/FraunhoferIWES/foxes>`_ with the shared optimizer
interface provided by `iwopy <https://github.com/FraunhoferIWES/iwopy>`_. It
supports vectorized population evaluation and uses FOXES execution engines for
local or distributed calculations.

.. csv-table::
    :header: "Resource", "Link"
    :widths: 20, 80

    "Documentation", "https://fraunhoferiwes.github.io/foxes-opt"
    "Source code", "https://github.com/FraunhoferIWES/foxes-opt"
    "PyPI", "https://pypi.org/project/foxes-opt/"
    "conda-forge", "https://anaconda.org/conda-forge/foxes-opt"
    "License", "`MIT <https://github.com/FraunhoferIWES/foxes-opt/blob/main/LICENSE>`_"

Requires Python 3.10 through 3.14.

Please report issues in the
`issue tracker <https://github.com/FraunhoferIWES/foxes-opt/issues>`_.

Contents
--------
    .. toctree::
        :maxdepth: 2

        citation

    .. toctree::
        :maxdepth: 2

        installation

    .. toctree::
        :maxdepth: 2

        parameter_files

    .. toctree::
        :maxdepth: 2

        examples

    .. toctree::
        :maxdepth: 1

        api

    .. toctree::
        :maxdepth: 1

        testing

    .. toctree::
        :maxdepth: 1

        CHANGELOG

Contributing
------------

#. Fork *foxes-opt* on *github*.
#. Create a branch (`git checkout -b new_branch`)
#. Commit your changes (`git commit -am "your awesome message"`)
#. Push to the branch (`git push origin new_branch`)
#. Create a pull request `here <https://github.com/FraunhoferIWES/foxes-opt/pulls>`_

Acknowledgements
----------------

Development of *FOXES Optimization* was supported by:

* BMWK projects *Smart Wind Farms* (0325851B), *GW-Wakes* (0325397B), and *X-Wakes* (03EE3008A)
* BMBF project *H2Digital* (03SF0635)
* Horizon Europe projects *FLOW* (101084205) and *AIRE* (101083716)
