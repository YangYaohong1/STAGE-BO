"""Adaptive epsilon-constraint selection rules used by STAGE-BO."""

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.transforms import normalize


def select_constraint_via_maxmin(
    posterior_pareto_front: torch.Tensor,
    train_obj: torch.Tensor,
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Select the next epsilon-constraint using the max-min rule."""
    if posterior_pareto_front.numel() == 0 or posterior_pareto_front.shape[0] == 0:
        non_dominated = is_non_dominated(train_obj)
        posterior_pareto_front = train_obj[non_dominated]
        if posterior_pareto_front.shape[0] == 0:
            posterior_pareto_front = train_obj

    bounds = torch.stack([train_obj.min(dim=-2).values, train_obj.max(dim=-2).values])
    posterior_norm = normalize(posterior_pareto_front, bounds=bounds)
    train_norm = normalize(train_obj, bounds=bounds)

    distances = torch.cdist(posterior_norm, train_norm)
    max_min_index = distances.min(dim=1).values.argmax().item()

    num_objectives = posterior_pareto_front.shape[-1]
    objective_index = iteration % num_objectives
    mask = torch.arange(num_objectives, device=posterior_pareto_front.device) != objective_index
    constraint_values = posterior_pareto_front[max_min_index, mask]
    selected_point = posterior_pareto_front[max_min_index]
    return constraint_values, selected_point, max_min_index, objective_index
