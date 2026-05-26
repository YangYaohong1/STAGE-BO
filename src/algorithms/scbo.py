import torch
import math
from dataclasses import dataclass
from torch import Tensor
from torch.quasirandom import SobolEngine
from moo_constraints.runtime import tkwargs


@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 2
    length_min: float = 0.5
    length_max: float = 2
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 5  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(2, **tkwargs) * torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([5.0 / self.batch_size, float(self.dim) / self.batch_size]))

def update_tr_length(state: ScboState) -> ScboState:
    state.length = 2
    return state

def get_best_index_for_batch(Y: Tensor, C: Tensor) -> int:
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return int(score.argmax().item())
    return int(C.clamp(min=0).sum(dim=-1).argmin().item())

def update_state(state: ScboState, Y_all: Tensor, C_all: Tensor, Y_next: Tensor, C_next: Tensor) -> ScboState:
    best_ind_batch = get_best_index_for_batch(Y=Y_next, C=C_next)
    y_next, c_next = Y_next[best_ind_batch], C_next[best_ind_batch]
    best_ind_all = get_best_index_for_batch(Y=Y_all, C=C_all)
    y_best, c_best = Y_all[best_ind_all], C_all[best_ind_all]
    state.best_value = y_best.item()
    state.best_constraint_values = c_best

    if (c_next <= 0).all():
        # At least one new candidate is feasible
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # No new candidate is feasible
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1

    return update_tr_length(state)

def generate_batch(
    state: ScboState,
    model,
    X: Tensor,
    Y: Tensor,
    C: Tensor,
    batch_size: int,
    n_candidates: int,
    constraint_model,
    sobol: SobolEngine,
):
    best_ind_list = []
    x_center_list = []
    y_center_list = []
    c_center_list = []
    tr_lb_list = []
    tr_ub_list = []
    X_cand_list = []

    for i in range(X.shape[0]):
        best_ind = get_best_index_for_batch(Y=Y[i], C=C[i])
        x_center = X[i, best_ind, :].clone()
        y_center = Y[i, best_ind].clone()
        c_center = C[i, best_ind].clone()
        tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)
        best_ind_list.append(best_ind)
        x_center_list.append(x_center)
        y_center_list.append(y_center)
        c_center_list.append(c_center)
        tr_lb_list.append(tr_lb)
        tr_ub_list.append(tr_ub) 

        dim = X.shape[-1]
        pert = sobol.draw(n_candidates).to(**tkwargs)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        if len(ind) > 0:
            if dim == 1:
                mask[ind, 0] = 1
            else:
                randint_indices = torch.randint(0, dim, size=(len(ind),), device=tkwargs["device"])
                mask[ind, randint_indices] = 1

        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask].to(X_cand.dtype)
        X_cand_list.append(X_cand)
    X_cand = torch.stack(X_cand_list).reshape(-1,dim)

    best_idx = get_best_index_for_batch(Y=torch.stack(y_center_list), C=torch.stack(c_center_list))
    X_next = x_center_list[best_idx].unsqueeze(0)
    return X_next, X_cand, x_center_list, state.length
