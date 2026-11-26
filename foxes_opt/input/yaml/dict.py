from typing import Any
from copy import deepcopy

from foxes.input.yaml import read_dict as foxes_read_dict
from foxes.input.yaml import run_outputs as foxes_run_output
from foxes.core import Algorithm, run_with_engine
from foxes.utils import Dict
from iwopy import LocalFD, create_optimizer
from iwopy.core import Optimizer

from foxes_opt.core import FarmConstraint, FarmObjective, FarmOptProblem
from foxes_opt.output import OptOutput


def read_dict(
    idict: Dict,
    *args: Any,
    verbosity: int | None = None,
    **kwargs: Any,
) -> tuple[Any, Any, Optimizer]:
    """
    Construct FOXES objects and an initialized iwopy optimizer.

    The optimizer mapping requires ``family`` and ``optimizer`` selectors,
    followed by explicit iwopy controls and flat native algorithm options.

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
    optimizer = create_optimizer(problem=problem, **sdict)
    optimizer.initialize()

    return algo, engine, optimizer


def run_outputs(
    idict: Dict,
    algo: Algorithm | None = None,
    farm_results: Any = None,
    opt_results: Any = None,
    with_engine: bool = False,
    results_storage: Dict | None = None,
    nofig: bool = False,
    verbosity: int | None = None,
) -> tuple[list[tuple[dict[str, Any], list[Any] | None]], Dict]:
    """Run constructor-defined optimization and FOXES outputs for one phase.

    Parameters
    ----------
    idict
        Configuration with constructor-defined ``outputs``. Only Matplotlib
        entries accept imperative ``functions`` lists.
    algo
        Algorithm executing FOXES and optimization output models.
    farm_results
        Supplied farm dataset, forwarded without recalculation.
    opt_results
        iwopy results injected into optimization-result writer constructors.
    with_engine
        Execute entries matching this phase. Models default to the engine
        phase; Matplotlib entries default to the non-engine phase.
    results_storage
        Shared result-reference mapping, or ``None`` to create one.
    nofig
        Suppress figures without disabling numerical result writers.
    verbosity
        Progress verbosity, where zero is silent.

    Returns
    -------
    outputs
        Ordered definition/result pairs; skipped entries have ``None`` results.
    results_storage
        Shared mapping for subsequent output entries and phases.

    Raises
    ------
    ValueError
        If an optimization writer lacks an algorithm or iwopy results, uses
        imperative functions, or supplies an invalid result label.
    """
    if results_storage is None:
        results_storage = Dict(_name="result_storage")
    outputs: list[tuple[dict[str, Any], list[Any] | None]] = []
    for entry in idict.get("outputs", []):
        if entry.get("output_type") in (
            "SingleObjResultsWriter",
            "MultiObjResultsWriter",
        ):
            definition = Dict(deepcopy(entry))
            if "functions" in definition:
                raise ValueError(
                    "Configure optimization writers through constructor options"
                )
            if definition.pop("with_engine", True) != with_engine:
                outputs.append((dict(entry), None))
                continue
            result = _run_opt_output(definition, algo, opt_results, results_storage)
            outputs.append((dict(entry), [result]))
        else:
            result, results_storage = foxes_run_output(
                Dict({"outputs": [entry]}),
                algo,
                farm_results=farm_results,
                with_engine=with_engine,
                results_storage=results_storage,
                nofig=nofig,
                verbosity=verbosity,
            )
            outputs.extend(result)
    return outputs, results_storage


def _run_opt_output(
    definition: Dict,
    algo: Algorithm | None,
    opt_results: Any,
    storage: Dict,
) -> Any:
    """Execute one optimization writer without scientific worker chunks."""
    if algo is None or opt_results is None:
        raise ValueError(
            f"{definition['output_type']}: Algorithm and optimization results are required"
        )
    label = definition.pop("result_labels", None)
    if label is not None and (
        not isinstance(label, str)
        or len(label) < 2
        or not label.startswith("$")
        or any(char in label for char in "[],")
    ):
        raise ValueError("Optimization output result_labels must be a simple '$name'")
    output = OptOutput.new(opt_results=opt_results, **definition)
    result = run_with_engine(algo.calc_outputs, outputs=output)
    if label is not None:
        storage[label] = result
    return result


def run_dict(
    idict: Dict,
    *args: Any,
    nofig: bool = False,
    verbosity: int | None = None,
    **kwargs: Any,
) -> tuple[Any, tuple[Any, ...]]:
    """Solve and execute constructor-defined outputs from a parameter dictionary.

    Engine-phase outputs run before non-engine outputs, sharing stored result
    references. The returned entries retain input order, not execution order.

    Parameters
    ----------
    idict
        The input parameter dictionary
    args
        Additional parameters for foxes.input.run_dict
    nofig
        Suppress all figure creation and display while retaining numerical
        writers, overriding output settings in ``idict``.
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
        One ``(definition, results)`` pair per configured output. Model results
        are single-element lists; Matplotlib results have one value per call.
        Outputs skipped because solving is disabled have ``None`` results.

    """

    def _print(*args: Any, level: int = 1, **kwargs: Any) -> None:
        if verbosity is None or verbosity >= level:
            print(*args, **kwargs)

    # read components:
    algo, engine, optimizer = read_dict(idict, *args, verbosity=verbosity, **kwargs)

    if verbosity is None or verbosity >= 0:
        optimizer.print_info()

    # run optimizer:
    rdict = idict.get_item("solve", Dict(_name=idict.name + ".solve"))
    results_storage = None
    out_w: list[Any] = [(dict(entry), None) for entry in idict.get("outputs", [])]
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
                with_engine=True,
                nofig=nofig,
                results_storage=results_storage,
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
    out_wo, _ = run_outputs(
        idict,
        algo,
        farm_results,
        opt_results,
        with_engine=False,
        nofig=nofig,
        results_storage=results_storage,
        verbosity=verbosity,
    )

    # combine outputs:
    out = tuple(
        a if a[1] is not None else b for a, b in zip(out_w, out_wo, strict=True)
    )

    return opt_results, out
