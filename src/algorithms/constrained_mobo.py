import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch import settings
from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination
from botorch.utils.multi_objective.pareto import is_non_dominated

from moo_constraints.runtime import tkwargs

class ConstrainedMOBO:
    def __init__(self, problem, noise_se, seed=1):
        self.problem = problem
        self.noise_se = noise_se
        self.standard_bounds = torch.zeros(2, self.problem.dim)
        self.standard_bounds[1] = 1
        self.seed = seed

    def generate_initial_data(self, n=4):
        train_x = draw_sobol_samples(bounds=self.problem.bounds, n=n, q=1, seed=self.seed).squeeze(1)
        train_obj_true = self.problem(train_x)
        train_obj = train_obj_true + torch.randn_like(train_obj_true) * self.noise_se
        train_con = self.problem.evaluate_slack(train_x)
        train_x = normalize(train_x, self.problem.bounds)
        return train_x, torch.round(train_obj, decimals=2), torch.round(train_con, decimals=2)

    def get_true_pareto_constrained_nsga2(
        self, 
        population_size=1000, 
        max_gen=100):
        """Get the true Pareto front for Constrained MOO."""
        tkwargs = {
            "dtype": self.problem.ref_point.dtype,
            "device": self.problem.ref_point.device,
        }
        
        # Increase population for higher dimensions/objectives
        if self.problem.num_objectives >= 3:
            population_size = 2000

        # 1. Define the Wrapper Problem
        class ConstrainedPymooProblem(Problem):
            def __init__(self, problem_instance):
                self.problem = problem_instance
                super().__init__(
                    n_var=self.problem.dim,
                    n_obj=self.problem.num_objectives,
                    n_ieq_constr=self.problem.num_constraints,
                    type_var=torch.double,
                )
                self.xl = torch.zeros(self.problem.dim).numpy()
                self.xu = torch.ones(self.problem.dim).numpy()

            def _evaluate(self, x, out, *args, **kwargs):
                X = torch.from_numpy(x).to(**tkwargs)
                X_unnormalized = unnormalize(X, self.problem.bounds)
                
                y = self.problem(X_unnormalized)
                out["F"] = -y.cpu().numpy()
                if hasattr(self.problem, "evaluate_slack"):
                    slack = self.problem.evaluate_slack(X_unnormalized)
                    out["G"] = -slack.cpu().numpy() 
                else:
                    raise NotImplementedError("Problem must implement evaluate_slack or similar.")

        pymoo_problem = ConstrainedPymooProblem(problem_instance=self.problem)

        algorithm = NSGA2(
            pop_size=population_size,
            eliminate_duplicates=True,
        )
        
        res = minimize(
            pymoo_problem,
            algorithm,
            termination=MaximumGenerationTermination(max_gen),
            seed=0,
            verbose=False,
        )

        if res.X is None:
            return torch.empty(0, self.problem.num_objectives, **tkwargs), torch.empty(0, self.problem.dim, **tkwargs)

        X_pymoo = torch.tensor(res.X, **tkwargs)
        X_true_design = unnormalize(X_pymoo, self.problem.bounds)
        
        Y_true = self.problem(X_true_design)
        slack_true = self.problem.evaluate_slack(X_true_design)
        feas_mask = (slack_true >= 0).all(dim=-1)
        if not feas_mask.any():
            return torch.empty(0, self.problem.num_objectives, **tkwargs), torch.empty(0, self.problem.dim, **tkwargs)

        Y_feasible = Y_true[feas_mask]
        X_feasible = X_true_design[feas_mask]

        pareto_mask = is_non_dominated(Y_feasible)
        
        return Y_feasible[pareto_mask], X_feasible[pareto_mask]

    def get_pareto_constrained_nsga2(
        self,
        model,
        ThompsonSampling=False,
        max_gen=100,
        seed=42,
    ):
        tkwargs = {
            "dtype": self.problem.ref_point.dtype,
            "device": self.problem.ref_point.device,
        }
        population_size = 300
        n_obj = self.problem.num_objectives
        if ThompsonSampling:
            torch.manual_seed(seed)
            sample_path = get_matheron_path_model(model=model, sample_shape=torch.Size([1]))

        empty_pf = torch.empty(0, n_obj, **tkwargs)
        empty_preds = torch.empty(0, n_obj + self.problem.num_constraints, **tkwargs)
        empty_x = torch.empty(0, self.problem.dim, **tkwargs)

        class ConstrainedPosteriorMeanPymooProblem(Problem):
            def __init__(self, problem_instance):
                self.problem = problem_instance
                self.n_obj = n_obj
                super().__init__(
                    n_var=self.problem.dim,
                    n_obj=self.n_obj,
                    n_ieq_constr=self.problem.num_constraints,
                    type_var=torch.double,
                )
                self.xl = torch.zeros(self.problem.dim).numpy()
                self.xu = torch.ones(self.problem.dim).numpy()

            def _evaluate(self, x, out, *args, **kwargs):
                X = torch.from_numpy(x).to(**tkwargs)
                with torch.no_grad():
                    with settings.cholesky_max_tries(9):
                        if ThompsonSampling:
                            y = torch.round(sample_path(X.unsqueeze(-2)).squeeze(-1).squeeze(-1), decimals=2)
                        else:
                            y = model.posterior(X.unsqueeze(-2)).mean.squeeze(-2)
                if y.ndim == 3:
                    if y.shape[0] == 1:
                        y = y.squeeze(0)
                    elif y.shape[1] == 1:
                        y = y.squeeze(1)

                if y.ndim != 2:
                    raise RuntimeError(f"Expected posterior tensor to be 2D, got shape {tuple(y.shape)}")

                out["F"] = -y[:, 0:self.n_obj].cpu().numpy()
                out["G"] = -y[:, self.n_obj:].cpu().numpy()

        pymoo_problem = ConstrainedPosteriorMeanPymooProblem(problem_instance=self.problem)
        algorithm = NSGA2(
            pop_size=population_size,
            eliminate_duplicates=True,
        )
        res = minimize(
            pymoo_problem,
            algorithm,
            termination=MaximumGenerationTermination(max_gen),
            seed=0,
            verbose=False,
        )

        if res.X is None:
            if ThompsonSampling:
                return empty_pf, empty_preds, empty_x
            return empty_pf, empty_preds, empty_x
        
        X = torch.tensor(
            res.X,
            **tkwargs,
        )
        with torch.no_grad():
            if ThompsonSampling:
                preds = torch.round(sample_path(X.unsqueeze(-2)).squeeze(-2), decimals=2)
            else:
                preds = model.posterior(X.unsqueeze(-2)).mean.squeeze(-2)

        if preds.ndim == 3:
            if preds.shape[0] == 1:
                preds = preds.squeeze(0)
            elif preds.shape[1] == 1:
                preds = preds.squeeze(1)
        if preds.ndim != 2:
            raise RuntimeError(f"Expected posterior tensor to be 2D, got shape {tuple(preds.shape)}")

        feas_mask = (preds[:, n_obj:] >= 0).all(dim=-1)
        if not feas_mask.any():
            if ThompsonSampling:
                return empty_pf, empty_preds, empty_x
            return empty_pf, empty_preds, empty_x

        Y_feasible = preds[feas_mask, :]
        X_feasible = X[feas_mask]
        pareto_mask = is_non_dominated(Y_feasible[:, :n_obj])
        return self.problem(unnormalize(X_feasible[pareto_mask], self.problem.bounds)), Y_feasible[pareto_mask], X_feasible[pareto_mask]
