import foxes
import foxes.variables as FV
import numpy as np
from iwopy import create_optimizer

from foxes_opt.constraints import FarmBoundaryConstraint, MinDistConstraint
from foxes_opt.objectives import MaxFarmPower
from foxes_opt.problems.layout import FarmLayoutOptProblem


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

    solver = create_optimizer(
        family="pymoo",
        optimizer="GA",
        problem=problem,
        vectorize=True,
        pop_size=100,
        seed=42,
        termination="default",
        max_iterations=100,
        ftol=1e-6,
        xtol=1e-6,
    )
    solver.initialize()
    solver.print_info()

    engine = foxes.Engine.new(
        engine_type="process",
        n_procs=4,
        verbosity=0,
    )

    with engine:
        results = solver.solve()
        solver.finalize(results)

    print()
    print(results)

    farm_results = results.problem_results
    wke = 1 - farm_results[FV.P].values / farm_results[FV.AMB_P].values
    print("Wake losses:", wke)

    assert np.all(wke < 2e-4)


if __name__ == "__main__":
    test()
