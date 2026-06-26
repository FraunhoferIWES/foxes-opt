import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from iwopy.interfaces.pymoo import Optimizer_pymoo

import foxes
import foxes.variables as FV
from foxes_opt.constraints import FarmBoundaryConstraint, MinDistConstraint
from foxes_opt.objectives import MaxFarmPower
from foxes_opt.problems.layout import FarmLayoutOptProblem


def default_file_pattern():
    """Return the default NetCDF file pattern for this example."""
    return Path(__file__).resolve().parent / "data" / "states_*.nc"


def build_boundary():
    """Create the optimization boundary used by the example."""
    return (
        foxes.utils.geom2d.ClosedPolygon(
            np.array([[0, 0], [0, 1200], [1000, 800], [900, -200]], dtype=np.float64)
        )
        + foxes.utils.geom2d.ClosedPolygon(
            np.array([[500, 0], [500, 1500], [1000, 1500], [1000, 0]], dtype=np.float64)
        )
        - foxes.utils.geom2d.Circle([-100.0, -100.0], 700)
    )


def build_states(file_pattern, ti, rho, load_mode="preload", grid_point_plot=None):
    """Build FieldData states from one or many NetCDF files."""
    return foxes.input.states.FieldData(
        str(file_pattern),
        states_coord="state",
        x_coord="x",
        y_coord="y",
        h_coord=None,
        time_format=None,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2ncvar={FV.WS: "ws", FV.WD: "wd"},
        fixed_vars={FV.RHO: rho, FV.TI: ti},
        load_mode=load_mode,
        interp_pars=dict(bounds_error=False),
        grid_point_plot=grid_point_plot,
    )


def build_problem(
    file_pattern,
    turbine_file,
    n_t,
    rotor,
    wakes,
    pwakes,
    ti,
    rho,
    min_dist=None,
    load_mode="preload",
    grid_point_plot=None,
    verbosity=1,
):
    """Create the farm layout optimization problem for this example."""
    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(turbine_file)
    mbook.turbine_types[ttype.name] = ttype

    farm = foxes.WindFarm(boundary=build_boundary())
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.array([500.0, 500.0]),
        xy_step=np.array([50.0, 50.0]),
        n_turbines=n_t,
        turbine_models=[ttype.name],
    )

    states = build_states(
        file_pattern,
        ti=ti,
        rho=rho,
        load_mode=load_mode,
        grid_point_plot=grid_point_plot,
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model=rotor,
        wake_models=wakes,
        wake_frame="rotor_wd",
        partial_wakes=pwakes,
        mbook=mbook,
        verbosity=verbosity,
    )

    problem = FarmLayoutOptProblem("layout_field_data", algo)
    problem.add_objective(MaxFarmPower(problem))
    problem.add_constraint(FarmBoundaryConstraint(problem))
    if min_dist is not None:
        problem.add_constraint(
            MinDistConstraint(problem, min_dist=min_dist, min_dist_unit="D")
        )
    problem.initialize()

    return farm, algo, problem


def build_solver(problem, opt_algo, n_pop, n_gen, vectorize=False, seed=13):
    """Create the pymoo optimizer for the example problem."""
    solver = Optimizer_pymoo(
        problem,
        problem_pars=dict(vectorize=vectorize),
        algo_pars=dict(type=opt_algo, pop_size=n_pop, seed=seed),
        setup_pars=dict(),
        term_pars=dict(type="default", n_max_gen=n_gen, ftol=1e-6, xtol=1e-6),
    )
    solver.initialize()
    return solver


def parse_args():
    """Parse command line arguments for the example."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nt", "--n_t", help="The number of turbines", type=int, default=8
    )
    parser.add_argument(
        "-t",
        "--turbine_file",
        help="The P-ct-curve csv file (path or static)",
        default="NREL-5MW-D126-H90.csv",
    )
    parser.add_argument(
        "-f",
        "--file_pattern",
        help="The NetCDF file pattern for the heterogeneous states",
        default=str(default_file_pattern()),
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
    parser.add_argument("--ti", help="The TI value", type=float, default=0.06)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument(
        "-d",
        "--min_dist",
        help="Minimal turbine distance in unit D",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "-lm",
        "--load_mode",
        help="The state data load mode",
        default="preload",
    )
    parser.add_argument(
        "-gpp",
        "--grid_point_plot",
        help="Optional output path for the selected FieldData grid point figure",
        default=None,
    )
    parser.add_argument(
        "-A", "--opt_algo", help="The pymoo algorithm name", default="GA"
    )
    parser.add_argument(
        "-P", "--n_pop", help="The population size", type=int, default=40
    )
    parser.add_argument(
        "-G", "--n_gen", help="The number of generations", type=int, default=100
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
    parser.add_argument(
        "-nf", "--nofig", help="Do not show figures", action="store_true"
    )
    return parser.parse_args()


def main():
    """Run the layout optimization example with heterogeneous FieldData states."""
    args = parse_args()

    farm, algo, problem = build_problem(
        file_pattern=args.file_pattern,
        turbine_file=args.turbine_file,
        n_t=args.n_t,
        rotor=args.rotor,
        wakes=args.wakes,
        pwakes=args.pwakes,
        ti=args.ti,
        rho=args.rho,
        min_dist=args.min_dist,
        load_mode=args.load_mode,
        grid_point_plot=args.grid_point_plot,
    )
    solver = build_solver(
        problem,
        opt_algo=args.opt_algo,
        n_pop=args.n_pop,
        n_gen=args.n_gen,
        vectorize=not args.no_pop,
    )
    solver.print_info()

    if not args.nofig:
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

        plot_data = None
        if not args.nofig and results.problem_results is not None:
            o = foxes.output.FlowPlots2D(algo, results.problem_results)
            p_min = np.array([-100.0, -350.0])
            p_max = np.array([1100.0, 1600.0])
            plot_data = o.get_mean_data_xy(
                FV.WS,
                resolution=20,
                xmin=p_min[0],
                xmax=p_max[0],
                ymin=p_min[1],
                ymax=p_max[1],
            )

    if not args.nofig and plot_data is not None:
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        foxes.output.FarmLayoutOutput(farm).get_figure(fig=fig, ax=axs[0])
        fig = o.get_mean_fig_xy(plot_data, fig=fig, ax=axs[1])

        dpars = dict(alpha=0.6, zorder=10, p_min=p_min, p_max=p_max)
        farm.boundary.add_to_figure(
            axs[1], fill_mode="outside_white", pars_distance=dpars
        )

        plt.show()
        plt.close(fig)
    elif not args.nofig:
        print(
            "No plot data available because the optimizer did not return farm results."
        )


if __name__ == "__main__":
    main()
