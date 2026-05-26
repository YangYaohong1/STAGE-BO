import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import gpytorch
import numpy as np
import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

from moo_constraints.algorithms.constrained_mobo import ConstrainedMOBO
from moo_constraints.algorithms.constraint_selection import select_constraint_via_maxmin
from moo_constraints.algorithms.metrics import Metrics
from moo_constraints.algorithms.mobo import PreferenceMOBO
from moo_constraints.algorithms.scbo import ScboState, generate_batch, update_state
from moo_constraints.config import MethodConfig
from moo_constraints.models.gp import fit_model_list
from moo_constraints.runtime import tkwargs


class MethodSolver:
    def __init__(self, problem, config: MethodConfig):
        self.problem = problem
        self.config = config
        self.config.validate()
        self.noise_se = torch.tensor([config.noise] * self.problem.num_objectives, **tkwargs)
        self.preference_mobo = PreferenceMOBO(problem, self.noise_se, seed=config.seed)
        self.constrained_mobo = ConstrainedMOBO(problem, self.noise_se, seed=config.seed)
        self.output_dir = self._make_output_dir(config.output_dir)
        self.true_pareto_front, self.true_pareto_x = self._get_true_pareto_front()

    def solve(self) -> dict:
        if self.config.mode == "constrained":
            train_x, train_obj, train_con = self.constrained_mobo.generate_initial_data(
                n=self.config.initial_points or 2 * (self.problem.dim + 1)
            )
        else:
            train_x, train_obj, _ = self.preference_mobo.generate_initial_data(
                n=self.config.initial_points or 2 * (self.problem.dim + 1)
            )
            train_con = None

        history = []
        state = ScboState(self.problem.dim, batch_size=1)
        sobol = SobolEngine(self.problem.dim, scramble=True, seed=self.config.seed)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for iteration in range(1, self.config.steps + 1):
            observed_front, _ = self._get_observed_front(train_x, train_obj, train_con)
            metrics = self._compute_metrics(observed_front)
            posterior_model = self._fit_posterior_model(train_x, train_obj, train_con)

            posterior_mean_list = []
            posterior_x_list = []
            for sample_idx in range(self.config.num_ts_samples):
                if self.config.mode == "constrained":
                    _, post_mean, post_x = self.constrained_mobo.get_pareto_constrained_nsga2(
                        posterior_model,
                        ThompsonSampling=self.config.thompson_sampling,
                        seed=sample_idx,
                    )
                else:
                    _, _, post_mean, post_x = self.preference_mobo.get_pareto_nsga2(
                        posterior_model,
                        ThompsonSampling=self.config.thompson_sampling,
                        seed=sample_idx,
                        preference_region=self.config.preference_region,
                    )
                posterior_mean_list.append(post_mean)
                posterior_x_list.append(post_x)

            stacked_posterior_mean = torch.stack(posterior_mean_list)
            stacked_posterior_x = torch.stack(posterior_x_list)
            constraint_values, _, _, objective_index = select_constraint_via_maxmin(
                posterior_pareto_front=stacked_posterior_mean[0][:, : self.problem.num_objectives],
                train_obj=train_obj,
                iteration=iteration,
            )
            if self.config.num_ts_samples > 1:
                bounds_list = []
                for posterior_sample in stacked_posterior_mean:
                    sample_bounds, _, _, objective_index = select_constraint_via_maxmin(
                        posterior_sample[:, : self.problem.num_objectives],
                        train_obj,
                        iteration,
                    )
                    bounds_list.append(sample_bounds)
                constraint_values = torch.mean(torch.stack(bounds_list), dim=0)

            aux_targets = self._build_aux_targets(train_obj, train_con, objective_index)
            full_bounds = self._augment_bounds_for_constraints(constraint_values, train_con)
            full_bounds = self._relax_bounds_if_needed(aux_targets, full_bounds)

            objective_train_y = train_obj[:, objective_index].unsqueeze(-1)
            objective_model = fit_model_list(train_x, objective_train_y)
            constraint_model = fit_model_list(train_x, full_bounds - aux_targets)

            if iteration > 1:
                state = update_state(
                    state=state,
                    Y_all=objective_train_y,
                    C_all=full_bounds - aux_targets,
                    Y_next=objective_train_y,
                    C_next=full_bounds - aux_targets,
                )

            posterior_aux = self._extract_aux_outputs(stacked_posterior_mean, objective_index)
            posterior_scalar_y = self._posterior_objective_values(stacked_posterior_mean, objective_index)
            with gpytorch.settings.cholesky_jitter(1e-3):
                _, _, centers, length = generate_batch(
                    state=state,
                    model=objective_model,
                    X=stacked_posterior_x,
                    Y=posterior_scalar_y.unsqueeze(-1),
                    C=full_bounds - posterior_aux,
                    batch_size=1,
                    n_candidates=2000,
                    constraint_model=constraint_model,
                    sobol=sobol,
                )

            x_next = self._optimize_inner_acquisition(
                posterior_model=posterior_model,
                train_x=train_x,
                objective_index=objective_index,
                full_bounds=full_bounds,
                center=centers[0],
                length=length,
            )

            new_obj_true = torch.round(self.problem(unnormalize(x_next, self.problem.bounds)), decimals=2)
            new_obj = torch.round(new_obj_true + torch.randn_like(new_obj_true) * self.noise_se, decimals=2)
            train_x = torch.cat([train_x, x_next], dim=0)
            train_obj = torch.cat([train_obj, new_obj], dim=0)

            if self.config.mode == "constrained":
                new_con = torch.round(self.problem.evaluate_slack(unnormalize(x_next, self.problem.bounds)), decimals=2)
                train_con = torch.cat([train_con, new_con], dim=0)

            history.append(
                {
                    "iteration": iteration,
                    "objective_index": int(objective_index),
                    "constraint_values": [float(value) for value in full_bounds.detach().cpu().tolist()],
                    **metrics,
                }
            )

        self._save_outputs(
            timestamp=timestamp,
            train_x=train_x,
            train_obj=train_obj,
            train_con=train_con,
            history=history,
        )
        return {
            "output_dir": str(self.output_dir),
            "iterations": len(history),
            "final_metrics": history[-1] if history else {},
        }

    def _make_output_dir(self, root: Path) -> Path:
        output_dir = Path(root) / self.config.mode / self.config.problem / f"seed_{self.config.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _get_true_pareto_front(self):
        if self.config.mode == "constrained":
            return self.constrained_mobo.get_true_pareto_constrained_nsga2()

        pareto_front, pareto_x = self.preference_mobo.get_true_pareto_nsga2()
        if self.config.preference_region is not None:
            region = torch.tensor(self.config.preference_region, **tkwargs)
            mask = (pareto_front >= region[0]).all(dim=-1) | (pareto_front <= region[1]).all(dim=-1)
            filtered_front = pareto_front[mask]
            filtered_x = pareto_x[mask]
            if filtered_front.shape[0] > 0:
                pareto_front = filtered_front
                pareto_x = filtered_x
                self.problem.ref_point = pareto_front.min(dim=0).values

        if self.config.ref_point is not None:
            self.problem.ref_point = torch.tensor(self.config.ref_point, **tkwargs)
        return pareto_front, pareto_x

    def _fit_posterior_model(self, train_x, train_obj, train_con):
        train_y = train_obj if train_con is None else torch.cat([train_obj, train_con], dim=-1)
        return fit_model_list(train_x, train_y)

    def _get_observed_front(self, train_x, train_obj, train_con):
        if train_con is None:
            mask = is_non_dominated(train_obj)
            return train_obj[mask], train_x[mask]
        feasible = (train_con >= 0).all(dim=-1)
        feasible_obj = train_obj[feasible]
        feasible_x = train_x[feasible]
        if feasible_obj.shape[0] == 0:
            return (
                torch.empty(0, self.problem.num_objectives, **tkwargs),
                torch.empty(0, self.problem.dim, **tkwargs),
            )
        mask = is_non_dominated(feasible_obj)
        return feasible_obj[mask], feasible_x[mask]

    def _compute_metrics(self, observed_front):
        if observed_front.shape[0] == 0:
            return {
                "hv": 0.0,
                "igd": 10.0,
                "igd_plus": 10.0,
                "fill_distance": 10.0,
                "spacing": 10.0,
            }
        hv = DominatedPartitioning(ref_point=self.problem.ref_point, Y=observed_front).compute_hypervolume().item()
        return {
            "hv": hv,
            "igd": Metrics.inverted_generational_distance(observed_front, self.true_pareto_front),
            "igd_plus": Metrics.igd_plus(observed_front, self.true_pareto_front),
            "fill_distance": Metrics.fill_distance(observed_front, self.true_pareto_front),
            "spacing": Metrics.spacing(observed_front),
        }

    def _build_aux_targets(self, train_obj, train_con, objective_index):
        objective_mask = torch.arange(train_obj.shape[-1], device=train_obj.device) != objective_index
        aux = train_obj[:, objective_mask]
        if train_con is not None:
            aux = torch.cat([aux, train_con], dim=-1)
        return aux

    def _augment_bounds_for_constraints(self, objective_bounds, train_con):
        if train_con is None:
            return objective_bounds
        return torch.cat([objective_bounds, torch.zeros(train_con.shape[-1], **tkwargs)])

    def _relax_bounds_if_needed(self, aux_targets, full_bounds):
        if not self.config.relax_constraints:
            return full_bounds
        feasible = aux_targets >= full_bounds
        if (~feasible.any(dim=0)).any():
            infeasible_idx = feasible.any(dim=0)
            full_bounds = full_bounds.clone()
            full_bounds[~infeasible_idx] = aux_targets[:, ~infeasible_idx].max(dim=0).values
        return full_bounds

    def _extract_aux_outputs(self, stacked_posterior_mean, objective_index):
        total_outputs = stacked_posterior_mean.shape[-1]
        keep_indices = [idx for idx in range(total_outputs) if idx != objective_index]
        return stacked_posterior_mean[:, :, keep_indices]

    def _posterior_objective_values(self, stacked_posterior_mean, objective_index):
        objective_values = stacked_posterior_mean[:, :, objective_index]
        if self.config.mode == "constrained":
            return objective_values
        aux = self._extract_aux_outputs(stacked_posterior_mean, objective_index)
        denom = aux.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8)
        return objective_values + self.config.epsilon * (aux / denom).sum(dim=-1)

    def _optimize_inner_acquisition(self, posterior_model, train_x, objective_index, full_bounds, center, length):
        total_outputs = self.problem.num_objectives + (
            self.problem.num_constraints if self.config.mode == "constrained" else 0
        )
        keep_indices = [idx for idx in range(total_outputs) if idx != objective_index]

        if self.config.mode == "constrained":
            objective = GenericMCObjective(lambda Z, X=None: Z[..., objective_index])
        else:
            mask = torch.tensor(keep_indices, device=train_x.device)

            def objective_fn(Z, X=None):
                aux = Z[..., mask]
                denom = aux.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8)
                return Z[..., objective_index] + self.config.epsilon * (aux / denom).sum(dim=-1)

            objective = GenericMCObjective(objective_fn)

        def constraint_callable(Z, X=None):
            constrained_values = full_bounds - Z[..., keep_indices]
            return constrained_values.max(dim=-1).values

        acq_func = qLogNoisyExpectedImprovement(
            model=posterior_model,
            X_baseline=train_x,
            constraints=[constraint_callable],
            objective=objective,
        )
        tr_lb = torch.clamp(center - length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(center + length / 2.0, 0.0, 1.0)
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=1,
            num_restarts=5,
            raw_samples=2048,
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidates.detach()

    def _save_outputs(self, timestamp, train_x, train_obj, train_con, history):
        np.savetxt(self.output_dir / f"{timestamp}_train_x.csv", train_x.detach().cpu().numpy(), delimiter=",")
        np.savetxt(self.output_dir / f"{timestamp}_train_obj.csv", train_obj.detach().cpu().numpy(), delimiter=",")
        np.savetxt(
            self.output_dir / f"{timestamp}_true_pareto_front.csv",
            self.true_pareto_front.detach().cpu().numpy(),
            delimiter=",",
        )
        if train_con is not None:
            np.savetxt(self.output_dir / f"{timestamp}_train_con.csv", train_con.detach().cpu().numpy(), delimiter=",")

        with (self.output_dir / f"{timestamp}_history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        with (self.output_dir / f"{timestamp}_config.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.config), handle, indent=2, default=str)
