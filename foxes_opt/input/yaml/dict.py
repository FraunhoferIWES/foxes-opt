from iwopy import LocalFD
from iwopy.core import Optimizer
from foxes.input.yaml import read_dict as foxes_read_dict
from foxes.input.yaml import run_outputs as foxes_run_output
from foxes.utils import Dict

from foxes_opt.core import FarmOptProblem, FarmObjective, FarmConstraint


def read_dict(idict, *args, verbosity=None, **kwargs):
    """
    Read dictionary input into foxes objects

    Parameters
    ----------
    idict: foxes.utils.Dict
        The input parameter dictionary
    args: tuple, optional
        Additional parameters for foxes.input.run_dict
    verbosity: int, optional
        Force a verbosity level, 0 = silent, overrules
        settings from idict
    kwargs: dict, optional
        Additional parameters for foxes.input.run_dict

    Returns
    -------
    algo: foxes.core.Algorithm
        The algorithm
    engine: foxes.core.Engine
        The engine, or None if not set
    optimizer: iwopy.core.Optimizer
        The optimization problem solver

    :group: input.yaml

    """

    def _print(*args, level=1, **kwargs):
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

    # extract data:
    jdict = idict.pop_item("optimization")

    # read base components:
    algo, engine = foxes_read_dict(idict, *args, verbosity=verbosity, **kwargs)
    if engine is not None:
        engine.verbosity = 0

    # create problem:
    _print("Creating problem")
    pdict = jdict.get_item("problem")
    ldict = pdict.pop("local_fd", None)
    odicts = [
        Dict(o, _name=f"{pdict.name}.objective{i}")
        for i, o in enumerate(pdict.pop_item("objectives"))
    ]
    cdicts = pdict.pop("constraints", [])
    cdicts = [
        Dict(c, _name=f"{pdict.name}.constraint{i}") for i, c in enumerate(cdicts)
    ]
    flist = [
        Dict(f, _name=f"{pdict.name}.function{i}")
        for i, f in enumerate(pdict.pop("functions", []))
    ]
    problem = FarmOptProblem.new(algo=algo, **pdict)
    for fdict in flist:
        fname = fdict.pop_item("name")
        _print(f"  - {fname}")
        f = getattr(problem, fname)
        f(**fdict)
    for odict in odicts:
        _print(f"  Adding objective: {odict.get_item('objective_type')}")
        o = FarmObjective.new(problem=problem, **odict)
        problem.add_objective(o)
    for cdict in cdicts:
        _print(f"  Adding constraint: {odict.get_item('constraint_type')}")
        c = FarmConstraint.new(problem=problem, **cdict)
        problem.add_constraint(c)
    if ldict is not None:
        _print("Adding local finite differences")
        problem0 = problem
        problem = LocalFD(problem0, **ldict)
    problem.initialize()

    # create solver:
    _print("Creating optimizer")
    sdict = jdict.get_item("optimizer")
    optimizer = Optimizer.new(problem=problem, **sdict)
    optimizer.initialize()

    return algo, engine, optimizer


def run_outputs(
    idict,
    algo=None,
    farm_results=None,
    opt_results=None,
    extra_sig={},
    **kwargs,
):
    """
    Run outputs from dict.

    Parameters
    ----------
    idict: foxes.utils.Dict
        The input parameter dictionary
    algo: foxes.core.Algorithm, optional
        The algorithm
    farm_results: xarray.Dataset, optional
        The farm results
    opt_results: iwopy.core.SingleObjOptResults or iwopy.core.MultiObjOptResults, optional
        The optimization results
    extra_sig: dict
        Extra function signature check, sets
        arguments (key) with data (value)
    kwargs: dict, optional
        Additional parameters for foxes_run_output

    Returns
    -------
    outputs: list of tuple
        For each output enty, a tuple (dict, results),
        where results is a tuple that represents one
        entry per function call

    :group: input.yaml

    """
    extra_sig.update({"opt_results": opt_results})

    out = foxes_run_output(
        idict,
        algo,
        farm_results=farm_results,
        point_results=None,
        extra_sig=extra_sig,
        **kwargs,
    )

    return out


def run_dict(idict, *args, extra_sig={}, verbosity=None, **kwargs):
    """
    Run from a dictionary type parameter file.

    Parameters
    ----------
    idict: foxes.utils.Dict
        The input parameter dictionary
    args: tuple, optional
        Additional parameters for foxes.input.run_dict
    extra_sig: dict
        Extra function signature check, sets
        arguments (key) with data (value)
    verbosity: int, optional
        Force a verbosity level, 0 = silent, overrules
        settings from idict
    kwargs: dict, optional
        Additional parameters for foxes.input.run_dict

    Returns
    -------
    opt_results: iwopy.core.SingleObjOptResults or iwopy.core.MultiObjOptResults
        The optimization results
    outputs: list of tuple
        For each output enty, a tuple (dict, results),
        where results is a tuple that represents one
        entry per function call

    """

    def _print(*args, level=1, **kwargs):
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

    # read components:
    algo, engine, optimizer = read_dict(idict, *args, verbosity=verbosity, **kwargs)

    if verbosity is None or verbosity >= 0:
        optimizer.print_info()

    # run optimizer:
    rdict = idict.get_item("solve", Dict(_name=idict.name + ".solve"))
    if rdict.pop_item("run", True):
        _print("Running optimizer")
        opt_results = optimizer.solve(**rdict)
        optimizer.finalize(opt_results)
        farm_results = opt_results.problem_results
    else:
        opt_results = None
        farm_results = None

    print()
    print(opt_results)
    print()

    # run outputs:
    out = run_outputs(
        idict, algo, farm_results, opt_results, extra_sig=extra_sig, verbosity=verbosity
    )

    # shutdown engine, if created above:
    if engine is not None:
        _print(f"Finalizing engine: {engine}")
        engine.finalize()

    return opt_results, out
