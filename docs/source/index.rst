Welcome to *foxes-opt*
======================

.. versionadded:: 0.7.0
    Compatibility with *foxes* version 1.7.1

The package *foxes-opt* provides optimization functionality for the
*Farm Optimization and eXtended yield Evaluation Software* `foxes <https://github.com/FraunhoferIWES/foxes>`_
and is based on the optimization interface `iwopy <https://github.com/FraunhoferIWES/iwopy>`_.

All three open-source Python packages *foxes*, *foxes-opt* and *iwopy* are provided and maintained by Fraunhofer IWES.

The calculation is fully vectorized and its fast performance is owed to `dask <https://www.dask.org/>`_.
Also the parallelization on local or remote clusters is enabled via `dask`. The wind farm
optimization capabilities invoke the `iwopy <https://github.com/FraunhoferIWES/iwopy>`_
package which as well supports vectorization.

Source code repository (and issue tracker):
    https://github.com/FraunhoferIWES/foxes-opt

Please report code issues under the github link above.

License
-------
    MIT_

.. _MIT: https://github.com/FraunhoferIWES/foxes-opt/blob/main/LICENSE

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

The development of *foxes* and its predecessors *flapFOAM* and *flappy* (internal - non public)
has been supported through multiple publicly funded research projects. We acknowledge in particular
the funding by the Federal Ministry of Economic Affairs and Climate Action (BMWK) through the p
rojects *Smart Wind Farms* (grant no. 0325851B), *GW-Wakes* (0325397B) and *X-Wakes* (03EE3008A)
as well as the funding by the Federal Ministry of Education and Research (BMBF) in the framework
of the project *H2Digital* (03SF0635). We furthermore acknowledge funding by the Horizon Europe
project FLOW (Atmospheric Flow, Loads and pOwer for Wind energy - grant id 101084205).
