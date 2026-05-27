"""Core STAGE-BO utilities for unconstrained and preference-based settings."""

import random

import numpy as np
import torch
from botorch.sampling.pathwise.posterior_samplers import get_matheron_path_model
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch import settings
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible NSGA-II runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PreferenceStageBO:
    """Helper methods for unconstrained and preference-based STAGE-BO."""
    def __init__(self, problem, noise_se, seed=1):
        self.problem = problem
        self.noise_se = noise_se
        self.seed = seed

    def generate_initial_data(self, n=4):
        """Draw Sobol initial points and evaluate noisy objectives."""
        train_x = draw_sobol_samples(bounds=self.problem.bounds, n=n, q=1, seed=self.seed).squeeze(1)
        train_obj_true = self.problem(train_x)
        train_obj = train_obj_true + torch.randn_like(train_obj_true) * self.noise_se
        train_x = normalize(train_x, self.problem.bounds)
        return train_x, torch.round(train_obj, decimals=2), torch.round(train_obj_true, decimals=2)

    def get_true_pareto_nsga2(self, population_size=500, max_gen=100):
        """Approximate the true Pareto front of the underlying test problem."""
        local_tkwargs = {
            "dtype": self.problem.ref_point.dtype,
            "device": self.problem.ref_point.device,
        }
        set_all_seeds(self.seed)
        if self.problem.num_objectives >= 3:
            population_size = 2000

        class TrueProblem(Problem):
            def __init__(self, problem_instance):
                self.problem = problem_instance
                super().__init__(
                    n_var=self.problem.dim,
                    n_obj=self.problem.num_objectives,
                    type_var=torch.double,
                )
                self.xl = torch.zeros(self.problem.dim).numpy()
                self.xu = torch.ones(self.problem.dim).numpy()

            def _evaluate(self, x, out, *args, **kwargs):
                X = torch.from_numpy(x).to(**local_tkwargs)
                y = self.problem(unnormalize(X, self.problem.bounds))
                out["F"] = -y.cpu().numpy()

        result = minimize(
            TrueProblem(self.problem),
            NSGA2(pop_size=population_size, eliminate_duplicates=True),
            termination=MaximumGenerationTermination(max_gen),
            seed=0,
            verbose=False,
        )
        X = torch.tensor(result.X, **local_tkwargs)
        y_true = self.problem(unnormalize(X, self.problem.bounds))
        pareto_mask = is_non_dominated(y_true)
        return y_true[pareto_mask], X[pareto_mask]

    def get_pareto_nsga2(
        self,
        model,
        ThompsonSampling=False,
        max_gen=50,
        seed=42,
        preference_region=None,
    ):
        """Extract a Pareto front from the model posterior or a Thompson sample."""
        local_tkwargs = {
            "dtype": self.problem.ref_point.dtype,
            "device": self.problem.ref_point.device,
        }
        population_size = 300
        sample_path = None
        if ThompsonSampling:
            torch.manual_seed(seed)
            sample_path = get_matheron_path_model(model=model, sample_shape=torch.Size([1]))

        class PosteriorProblem(Problem):
            def __init__(self, problem_instance):
                self.problem = problem_instance
                super().__init__(
                    n_var=self.problem.dim,
                    n_obj=self.problem.num_objectives,
                    type_var=torch.double,
                )
                self.xl = torch.zeros(self.problem.dim).numpy()
                self.xu = torch.ones(self.problem.dim).numpy()

            def _evaluate(self, x, out, *args, **kwargs):
                X = torch.from_numpy(x).to(**local_tkwargs)
                with torch.no_grad():
                    with settings.cholesky_max_tries(9):
                        if ThompsonSampling:
                            y = torch.round(sample_path(X.unsqueeze(-2)).squeeze(-1).squeeze(-1), decimals=2)
                        else:
                            y = model.posterior(X.unsqueeze(-2)).mean.squeeze(-2)
                out["F"] = -y.cpu().numpy()

        result = minimize(
            PosteriorProblem(self.problem),
            NSGA2(pop_size=population_size, eliminate_duplicates=True),
            termination=MaximumGenerationTermination(max_gen),
            seed=seed,
            verbose=False,
        )
        if result.X is None:
            empty_pf = torch.empty(0, self.problem.num_objectives, **local_tkwargs)
            empty_x = torch.empty(0, self.problem.dim, **local_tkwargs)
            return 0.0, empty_pf, empty_pf, empty_x

        X = torch.tensor(result.X, **local_tkwargs)
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

        if preference_region is not None:
            # Preference mode keeps Pareto points inside the region if possible,
            # then backs off to partial overlap, and finally to the full front.
            region = torch.tensor(preference_region, **local_tkwargs)
            feasible_mask = (preds >= region[0]).all(dim=-1) | (preds <= region[1]).all(dim=-1)
            preds_feasible = preds[feasible_mask]
            X_feasible = X[feasible_mask]
            if preds_feasible.shape[0] == 0:
                feasible_mask = (preds >= region[0]).any(dim=-1) & (preds <= region[1]).any(dim=-1)
                preds_feasible = preds[feasible_mask]
                X_feasible = X[feasible_mask]
                if preds_feasible.shape[0] == 0:
                    preds_feasible = preds
                    X_feasible = X
            pareto_mask = is_non_dominated(preds_feasible)
            X_pf = X_feasible[pareto_mask]
            preds_pf = preds_feasible[pareto_mask]
        else:
            pareto_mask = is_non_dominated(preds)
            X_pf = X[pareto_mask]
            preds_pf = preds[pareto_mask]

        Y_pf = self.problem(unnormalize(X_pf, self.problem.bounds))
        partitioning = FastNondominatedPartitioning(ref_point=self.problem.ref_point, Y=Y_pf)
        return partitioning.compute_hypervolume().item(), Y_pf, preds_pf, X_pf
