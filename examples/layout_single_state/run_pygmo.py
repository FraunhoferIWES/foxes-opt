import argparse

import foxes
import matplotlib.pyplot as plt
import numpy as np
from iwopy import LocalFD
from iwopy.interfaces.pygmo import Optimizer_pygmo

from foxes_opt.constraints import FarmBoundaryConstraint, MinDistConstraint
from foxes_opt.objectives import MaxFarmPower
from foxes_opt.problems.layout import FarmLayoutOptProblem


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nt", "--n_t", type=int, default=10)
    parser.add_argument("-t", "--turbine_file", default="NREL-5MW-D126-H90.csv")
    parser.add_argument("-r", "--rotor", default="centre")
    parser.add_argument("-w", "--wakes", default=["Bastankhah025_linear_k002"], nargs="+")
    parser.add_argument("-p", "--pwakes", default=None)
    parser.add_argument("--ws", type=float, default=9.0)
    parser.add_argument("--wd", type=float, default=270.0)
    parser.add_argument("--ti", type=float, default=0.08)
    parser.add_argument("--rho", type=float, default=1.225)
    parser.add_argument("-d", "--min_dist", type=float, default=None)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--fd_order", type=int, default=1)
    parser.add_argument("--no_fig", action="store_true")
    parser.add_argument("-e", "--engine", default="process")
    parser.add_argument("-n", "--n_cpus", type=int, default=None)
    parser.add_argument("-c", "--chunksize_states", type=int, default=None)
    parser.add_argument("-C", "--chunksize_points", type=int, default=None)
    args = parser.parse_args()

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(args.turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    boundary = foxes.utils.geom2d.Circle([0.0, 0.0], 1000.0)
    farm = foxes.WindFarm(boundary=boundary)
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.zeros(2),
        xy_step=np.array([50.0, 0.0]),
        n_turbines=args.n_t,
        turbine_models=["kTI_02", ttype.name],
    )
    states = foxes.input.states.SingleStateStates(
        ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    )
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame="rotor_wd",
        partial_wakes=args.pwakes,
        mbook=mbook,
        verbosity=0,
    )

    problem = FarmLayoutOptProblem("layout_opt", algo)
    problem.add_objective(MaxFarmPower(problem))
    problem.add_constraint(FarmBoundaryConstraint(problem, disc_inside=True, infer_vars=True))
    if args.min_dist is not None:
        problem.add_constraint(
            MinDistConstraint(problem, min_dist=args.min_dist, min_dist_unit="D", infer_vars=True)
        )
    problem = LocalFD(problem, deltas=0.1, fd_order=args.fd_order)
    problem.initialize()

    solver = Optimizer_pygmo(
        problem,
        problem_pars={"pop": False},
        algo_pars={
            "type": "ipopt",
            "tol": args.tol,
            "max_iter": args.max_iter,
        },
        setup_pars={"pop_size": 1},
    )
    solver.initialize()
    solver.print_info()

    if not args.no_fig:
        ax = foxes.output.FarmLayoutOutput(farm).get_figure()
        plt.show()
        plt.close(ax.get_figure())

    engine = foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=0,
    )
    with engine:
        results = solver.solve()
        solver.finalize(results)
        print()
        print(results)
        if not args.no_fig:
            output = foxes.output.FlowPlots2D(algo, results.problem_results)
            p_min = np.array([-1100.0, -1100.0])
            p_max = np.array([1100.0, 1100.0])
            plot_data = output.get_mean_data_xy(
                "WS", resolution=20, xmin=p_min[0], xmax=p_max[0],
                ymin=p_min[1], ymax=p_max[1]
            )

    if not args.no_fig:
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        foxes.output.FarmLayoutOutput(farm).get_figure(fig=fig, ax=axs[0], bargs={})
        output.get_mean_fig_xy(plot_data, fig=fig, ax=axs[1])
        farm.boundary.add_to_figure(
            axs[1], fill_mode="outside_white",
            pars_distance={"alpha": 0.6, "zorder": 10, "p_min": p_min, "p_max": p_max},
        )
        plt.show()
        plt.close(fig)
