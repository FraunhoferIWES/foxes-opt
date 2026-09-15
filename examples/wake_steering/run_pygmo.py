import argparse

import foxes
import foxes.variables as FV
import matplotlib.pyplot as plt
import numpy as np
from iwopy import LocalFD
from iwopy.interfaces.pygmo import Optimizer_pygmo

from foxes_opt.objectives import MaxFarmPower
from foxes_opt.problems import OptFarmVars


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nt", "--n_t", type=int, default=9)
    parser.add_argument("-t", "--turbine_file", default="NREL-5MW-D126-H90.csv")
    parser.add_argument("-r", "--rotor", default="centre")
    parser.add_argument(
        "-w", "--wakes",
        default=["CrespoHernandez_quadratic_ambka04", "Bastankhah2014_vector_ambka04"],
        nargs="+",
    )
    parser.add_argument("-f", "--frame", default="rotor_wd")
    parser.add_argument("-d", "--deflection", default="Jimenez")
    parser.add_argument("-m", "--tmodels", default=[], nargs="+")
    parser.add_argument("-p", "--pwakes", default=None)
    parser.add_argument("--ws", type=float, default=9.0)
    parser.add_argument("--wd", type=float, default=270.0)
    parser.add_argument("--ti", type=float, default=0.03)
    parser.add_argument("--rho", type=float, default=1.225)
    parser.add_argument("-md", "--min_dist", type=float, default=None)
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
    farm = foxes.WindFarm()
    n_side = int(np.sqrt(args.n_t) + 0.5)
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=np.array([500.0, 500.0]),
        step_vectors=np.array([[1300.0, 0], [200, 600.0]]),
        steps=(n_side, n_side),
        turbine_models=args.tmodels + ["opt_yawm", "yawm2yaw", ttype.name],
    )
    states = foxes.input.states.SingleStateStates(
        ws=args.ws, wd=args.wd, ti=args.ti, rho=args.rho
    )
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model=args.rotor,
        wake_models=args.wakes,
        wake_frame=args.frame,
        wake_deflection=args.deflection,
        partial_wakes=args.pwakes,
        mbook=mbook,
        verbosity=0,
    )

    problem = OptFarmVars("opt_yawm", algo)
    problem.add_var(FV.YAWM, float, 0.0, -40.0, 40.0, level="turbine")
    problem.add_objective(MaxFarmPower(problem))
    problem = LocalFD(problem, deltas=0.1, fd_order=args.fd_order)
    problem.initialize()

    solver = Optimizer_pygmo(
        problem,
        problem_pars={"pop": False},
        algo_pars={"type": "ipopt", "tol": args.tol, "max_iter": args.max_iter},
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
        print(results.problem_results.to_dataframe()[[FV.X, FV.Y, FV.AMB_WD, FV.REWS, FV.TI, FV.P, FV.YAWM]])
        if not args.no_fig:
            output = foxes.output.FlowPlots2D(algo, results.problem_results)
            plot_data = output.get_mean_data_xy("WS", resolution=10)

    if not args.no_fig:
        output.get_mean_fig_xy(plot_data)
        plt.show()
