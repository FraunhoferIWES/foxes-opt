import argparse
import gc

import foxes
import foxes.variables as FV
import matplotlib.pyplot as plt
import numpy as np
from iwopy import LocalFD, create_optimizer

from foxes_opt.constraints import FarmBoundaryConstraint, MinDistConstraint
from foxes_opt.objectives import MaxFarmPower
from foxes_opt.problems.layout import FarmLayoutOptProblem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nt", "--n_t", help="The number of turbines", type=int, default=10
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-s",
        "--states",
        help="The states input file (path or static)",
        default="wind_rose_bremen.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["Bastankhah025_quadratic_k002"],
        nargs="+",
    )
    parser.add_argument("-p", "--pwakes", help="The partial wakes model", default=None)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.04)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument(
        "-d",
        "--min_dist",
        help="Minimal turbine distance in unit D",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-O",
        "--fd_order",
        help="Finite difference derivative order",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-I",
        "--maxiter",
        help="Maximum number of IPOPT iterations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--tol",
        help="IPOPT convergence tolerance",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "-nop", "--no_pop", help="Switch off vectorization", action="store_true"
    )
    parser.add_argument("-e", "--engine", help="The engine", default="process")
    parser.add_argument(
        "-n", "--n_cpus", help="The number of cpus", default=None, type=int
    )
    parser.add_argument(
        "-c",
        "--chunksize_states",
        help="The chunk size for states",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-C",
        "--chunksize_points",
        help="The chunk size for points",
        default=None,
        type=int,
    )
    args = parser.parse_args()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    boundary = (
        foxes.utils.geom2d.ClosedPolygon(
            np.array([[0, 0], [0, 1200], [1000, 800], [900, -200]], dtype=np.float64)
        )
        + foxes.utils.geom2d.ClosedPolygon(
            np.array([[500, 0], [500, 1500], [1000, 1500], [1000, 0]], dtype=np.float64)
        )
        - foxes.utils.geom2d.Circle([-100.0, -100.0], 700)
    )

    farm = foxes.WindFarm(boundary=boundary)
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.array([500.0, 500.0]),
        xy_step=np.array([50.0, 50.0]),
        n_turbines=args.n_t,
        turbine_models=[ttype.name],
    )

    states = foxes.input.states.StatesTable(
        data_source=args.states,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
        fixed_vars={FV.RHO: args.rho, FV.TI: args.ti},
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes=args.pwakes,
        mbook=mbook,
        verbosity=1,
    )

    ax = foxes.core.run_with_engine(
        algo.calc_outputs,
        outputs=foxes.models.outputs.StatesRosePlotOutput(
            point=[0.0, 0.0, 100.0],
            wd_sectors=16,
            ws_var=FV.AMB_WS,
            ws_bins=[0, 3.5, 6, 10, 15, 20],
        ),
    )[0]
    plt.show()
    plt.close(ax.get_figure())
    del ax

    problem = FarmLayoutOptProblem("layout_opt", algo)
    problem.add_objective(MaxFarmPower(problem))
    problem.add_constraint(FarmBoundaryConstraint(problem))
    if args.min_dist is not None:
        problem.add_constraint(
            MinDistConstraint(problem, min_dist=args.min_dist, min_dist_unit="D")
        )
    problem = LocalFD(problem, deltas=10.0, fd_order=args.fd_order)
    problem.initialize()

    solver = create_optimizer(
        family="pygmo",
        optimizer="ipopt",
        problem=problem,
        vectorize=not args.no_pop,
        max_iterations=args.maxiter,
        print_level=5,
        tol=args.tol,
    )
    solver.initialize()
    solver.print_info()

    ax = foxes.core.run_with_engine(
        algo.calc_outputs,
        outputs=foxes.models.outputs.FarmLayoutOutput(variables=[], parameters=dict()),
    )
    plt.show()
    plt.close(ax.get_figure())
    del ax
    gc.collect()

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=0,
    )

    with engine:
        results = solver.solve(verbosity=0)
        solver.finalize(results)

        print()
        print(results)

        p_min = np.array([-100.0, -350.0])
        p_max = np.array([1100.0, 1600.0])
        flow_pars = dict(
            var="WS",
            grid_pars=dict(
                resolution=20,
                xmin=p_min[0],
                xmax=p_max[0],
                ymin=p_min[1],
                ymax=p_max[1],
            ),
        )

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    foxes.core.run_with_engine(
        algo.calc_outputs,
        outputs=foxes.models.outputs.FarmLayoutOutput(
            variables=[], parameters=dict(fig=fig, ax=axs[0])
        ),
    )

    with engine:
        fig = algo.calc_outputs(
            outputs=foxes.models.outputs.FlowPlots2D(
                **flow_pars, plot_pars=dict(fig=fig, ax=axs[1])
            ),
            farm_results=results.problem_results,
        )
    dpars = {"alpha": 0.6, "zorder": 10, "p_min": p_min, "p_max": p_max}
    farm.boundary.add_to_figure(axs[1], fill_mode="outside_white", pars_distance=dpars)

    plt.show()
    plt.close(fig)
