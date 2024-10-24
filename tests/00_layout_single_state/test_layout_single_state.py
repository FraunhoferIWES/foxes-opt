import numpy as np
from iwopy.interfaces.pymoo import Optimizer_pymoo

import foxes
from foxes_opt.problems.layout import FarmLayoutOptProblem
from foxes_opt.constraints import FarmBoundaryConstraint, MinDistConstraint
from foxes_opt.objectives import MaxFarmPower


def test():

    boundary = foxes.utils.geom2d.Circle([0.0, 0.0], 1000.0)

    farm = foxes.WindFarm(boundary=boundary)
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.zeros(2),
        xy_step=np.array([50.0, 0.0]),
        n_turbines=8,
        turbine_models=["NREL5MW"],
    )
    states = foxes.input.states.SingleStateStates(ws=9, wd=270, ti=0.08, rho=1.225)

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear_lim_k004"],
        verbosity=0,
    )

    problem = FarmLayoutOptProblem("layout_opt", algo)
    problem.add_objective(MaxFarmPower(problem))
    problem.add_constraint(FarmBoundaryConstraint(problem, disc_inside=True))
    problem.add_constraint(MinDistConstraint(problem, min_dist=3, min_dist_unit="D"))
    problem.initialize()

    solver = Optimizer_pymoo(
        problem,
        problem_pars=dict(
            vectorize=True,
        ),
        algo_pars=dict(
            type="GA",
            pop_size=100,
            seed=41,
        ),
        setup_pars=dict(),
        term_pars=dict(
            type="default",
            n_max_gen=100,
            ftol=1e-6,
            xtol=1e-6,
        ),
    )
    solver.initialize()
    solver.print_info()

    """
    ax = foxes.output.FarmLayoutOutput(farm).get_figure()
    plt.show()
    plt.close(ax.get_figure())
    """

    results = solver.solve()
    solver.finalize(results)

    print()
    print(results)

    assert np.abs(results.objs - 20148.46492534) < 1e-4

    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    foxes.output.FarmLayoutOutput(farm).get_figure(fig=fig, ax=axs[0])

    o = foxes.output.FlowPlots2D(algo, results.problem_results)
    p_min = np.array([-1100.0, -1100.0])
    p_max = np.array([1100.0, 1100.0])
    fig = o.get_mean_fig_xy(
        "WS",
        resolution=20,
        fig=fig,
        ax=axs[1],
        xmin=p_min[0],
        xmax=p_max[0],
        ymin=p_min[1],
        ymax=p_max[1],
    )
    
    dpars = dict(alpha=0.6, zorder=10, p_min=p_min, p_max=p_max)
    farm.boundary.add_to_figure(
        axs[1], fill_mode="outside_white", pars_distance=dpars
    )
    plt.show()
    plt.close(fig)
    """


if __name__ == "__main__":

    test()
