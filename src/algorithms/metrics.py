"""Quality indicators for comparing observed and reference Pareto fronts."""

import numpy as np
import torch
from numpy import mean, sqrt, sum
from numpy.linalg import norm


class Metrics:
    @staticmethod
    def inverted_generational_distance(pareto_front: torch.Tensor, reference_front: torch.Tensor) -> float:
        """Compute IGD from the observed Pareto front to the reference front."""
        rf_size = len(reference_front)
        igd = 0.0
        for point in reference_front:
            dist_min = min(norm(point - candidate) for candidate in pareto_front)
            igd += dist_min
        return float(igd / rf_size)

    @staticmethod
    def igd_plus(pareto_front: torch.Tensor, reference_front: torch.Tensor) -> float:
        """Compute IGD+ using the standard non-improving directional distance."""
        rf_size = len(reference_front)
        igd_plus = 0.0
        for point in reference_front:
            dist_min = min(norm(torch.clamp(point - candidate, min=1e-2)) for candidate in pareto_front)
            igd_plus += dist_min
        return float(igd_plus / rf_size)

    @staticmethod
    def fill_distance(pareto_front: torch.Tensor, reference_front: torch.Tensor) -> float:
        """Measure the largest gap from the reference front to the observed front."""
        distances = np.zeros(len(reference_front))
        for idx, point in enumerate(reference_front):
            distances[idx] = min(norm(candidate - point) for candidate in pareto_front)
        return float(max(distances))

    @staticmethod
    def spacing(pareto_front: torch.Tensor) -> float:
        """Measure how evenly spaced the observed Pareto points are."""
        pf_size = len(pareto_front)
        if pf_size <= 1:
            return 1.0
        distances = np.zeros(pf_size)
        for idx in range(pf_size):
            distances[idx] = min(
                norm(pareto_front[idx] - pareto_front[jdx])
                for jdx in range(pf_size)
                if idx != jdx
            )
        dist_mean = mean(distances)
        return float(sqrt(sum((distances - dist_mean) ** 2) / (pf_size - 1)))
