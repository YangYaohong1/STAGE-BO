"""Thin orchestration layer for running one configured experiment."""

from config import MethodConfig
from problems import get_test_problem
from solvers import MethodSolver


def run_experiment(config: MethodConfig) -> dict:
    """Instantiate the problem and solver, then execute the optimization loop."""
    config.validate()
    problem = get_test_problem(config.problem)
    solver = MethodSolver(problem, config)
    return solver.solve()
