from moo_constraints.config import MethodConfig
from moo_constraints.problems import get_test_problem
from moo_constraints.solvers import MethodSolver


def run_experiment(config: MethodConfig) -> dict:
    config.validate()
    problem = get_test_problem(config.problem)
    solver = MethodSolver(problem, config)
    return solver.solve()
