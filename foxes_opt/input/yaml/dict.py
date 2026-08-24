from typing import Any

from iwopy import LocalFD
from iwopy.core import Optimizer
from foxes.input.yaml import read_dict as foxes_read_dict
from foxes.input.yaml import run_outputs as foxes_run_output
from foxes.utils import Dict

from foxes_opt.core import FarmOptProblem, FarmObjective, FarmConstraint


def read_dict(
    idict: Dict,
    *args: Any,
    verbosity: int | None = None,
    **kwargs: Any,
) -> tuple[Any, Any, Optimizer]:
    """
    Read dictionary input into foxes objects

    Parameters
    ----------
    idict
        The input parameter dictionary
    args
        Additional parameters for foxes.input.run_dict
    verbosity
        Force a verbosity level, 0 = silent, overrules
        settings from idict
    kwargs
        Additional parameters for foxes.input.run_dict

    Returns
    -------
    algo
        The algorithm
    engine
        The engine, or None if not set
    optimizer
        The optimization problem solver


    """

    def _print(*args: Any, level: int = 1, **kwargs: Any) -> None:
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
    idict: Dict,
    algo: Any = None,
    farm_results: Any = None,
    opt_results: Any = None,
    extra_sig: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Run outputs from dict.

    Parameters
    ----------
    idict
        The input parameter dictionary
    algo
        The algorithm
    farm_results
        The farm results
    opt_results
        The optimization results
    extra_sig
        Extra function signature check, sets
        arguments (key) with data (value)
    kwargs
        Additional parameters for foxes_run_output

    Returns
    -------
    outputs
        For each output enty, a tuple (dict, results),
        where results is a tuple that represents one
        entry per function call


    """
    if extra_sig is None:
        extra_sig = {}
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


def run_dict(
    idict: Dict,
    *args: Any,
    extra_sig: dict[str, Any] | None = None,
    nofig: bool = False,
    verbosity: int | None = None,
    **kwargs: Any,
) -> tuple[Any, tuple[Any, ...]]:
    """
    Run from a dictionary type parameter file.

    Parameters
    ----------
    idict
        The input parameter dictionary
    args
        Additional parameters for foxes.input.run_dict
    extra_sig
        Extra function signature check, sets
        arguments (key) with data (value)
    nofig
        Do not show figures, overrules settings from idict
    verbosity
        Force a verbosity level, 0 = silent, overrules
        settings from idict
    kwargs
        Additional parameters for foxes.input.run_dict

    Returns
    -------
    opt_results
        The optimization results
    outputs
        For each output enty, a tuple (dict, results),
        where results is a tuple that represents one
        entry per function call

    """

    def _print(*args: Any, level: int = 1, **kwargs: Any) -> None:
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

    if extra_sig is None:
        extra_sig = {}

    # read components:
    algo, engine, optimizer = read_dict(idict, *args, verbosity=verbosity, **kwargs)

    if verbosity is None or verbosity >= 0:
        optimizer.print_info()

    # run optimizer:
    rdict = idict.get_item("solve", Dict(_name=idict.name + ".solve"))
    results_storage = None
    out_w: list[Any] = []
    if rdict.pop_item("run", True):
        _print("Running optimizer")
        with engine:
            opt_results = optimizer.solve(**rdict)
            optimizer.finalize(opt_results)
            farm_results = opt_results.problem_results

            # run outputs with engine:
            out_w, results_storage = run_outputs(
                idict,
                algo,
                farm_results,
                opt_results,
                extra_sig=extra_sig,
                with_engine=True,
                nofig=nofig,
                results_storage=results_storage,
                ret_results_storage=True,
                verbosity=verbosity,
            )
            out_w = list(out_w)

    else:
        opt_results = None
        farm_results = None

    print()
    print(opt_results)
    print()

    # run outputs w/o engine:
    out_wo = list(
        run_outputs(
            idict,
            algo,
            farm_results,
            opt_results,
            extra_sig=extra_sig,
            with_engine=False,
            nofig=nofig,
            results_storage=results_storage,
            ret_results_storage=True,
            verbosity=verbosity,
        ),
    )

    # combine outputs:
    out = tuple(a if a is not None else b for a, b in zip(out_w, out_wo))

    return opt_results, out
