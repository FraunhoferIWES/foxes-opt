import numpy as np
import argparse
import matplotlib.pyplot as plt
from iwopy import LocalFD
from iwopy.optimizers import GG

import foxes
from foxes_opt.problems import OptFarmVars
from foxes_opt.objectives import MaxFarmPower
import foxes.variables as FV

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nt", "--n_t", help="The number of turbines", type=int, default=9
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument(
        "-w",
        "--wakes",
        help="The wake models",
        default=["CrespoHernandez_quadratic", "Bastankhah2016_linear"],
        nargs="+",
    )
    parser.add_argument(
        "-m", "--tmodels", help="The turbine models", default=["kTI_02"], nargs="+"
    )
    parser.add_argument("-p", "--pwakes", help="The partial wakes model", default=None)
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.03)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument(
        "-d",
        "--min_dist",
        help="Minimal turbine distance in unit D",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-nop", "--no_pop", help="Switch off vectorization", action="store_true"
    )
    parser.add_argument(
        "-O",
        "--fd_order",
        help="Finite difference derivative order",
        type=int,
        default=1,
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

    farm = foxes.WindFarm()
    N = int(np.sqrt(args.n_t) + 0.5)
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=np.array([500.0, 500.0]),
        step_vectors=np.array([[1300.0, 0], [200, 600.0]]),
        steps=(N, N),
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
        wake_frame="yawed",
        partial_wakes=args.pwakes,
        mbook=mbook,
        verbosity=0,
    )

    with foxes.Engine.new(
        engine_type=args.engine,
        n_procs=args.n_cpus,
        chunk_size_states=args.chunksize_states,
        chunk_size_points=args.chunksize_points,
        verbosity=0,
    ):
        problem = OptFarmVars("opt_yawm", algo)
        problem.add_var(FV.YAWM, float, 0.0, -40.0, 40.0, level="turbine")
        problem.add_objective(MaxFarmPower(problem))
        problem.initialize()
        gproblem = LocalFD(problem, deltas=0.1, fd_order=args.fd_order)
        gproblem.initialize()

        solver = GG(
            gproblem,
            step_max=100.0,
            step_min=0.1,
            f_tol=1e-4,
            vectorized=not args.no_pop,
        )
        solver.initialize()
        solver.print_info()

        ax = foxes.output.FarmLayoutOutput(farm).get_figure()
        plt.show()
        plt.close(ax.get_figure())

        results = solver.solve()
        solver.finalize(results)

        print()
        print(results)

        fr = results.problem_results.to_dataframe()
        print(fr[[FV.X, FV.Y, FV.AMB_WD, FV.REWS, FV.TI, FV.P, FV.YAWM]])

        o = foxes.output.FlowPlots2D(algo, results.problem_results)
        fig = o.get_mean_fig_xy("WS", resolution=10)
        plt.show()
