"""Experiment configuration and validation for STAGE-BO runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MethodConfig:
    """User-facing configuration for a single STAGE-BO experiment."""
    problem: str
    mode: str
    steps: int
    seed: int = 1
    noise: float = 0.0
    initial_points: Optional[int] = None
    num_ts_samples: int = 1
    relax_constraints: bool = True
    preference_region: Optional[list[list[float]]] = None
    ref_point: Optional[list[float]] = None
    output_dir: Path = Path("outputs")

    constraint_strategy: str = "maxmin"
    inner_acquisition: str = "cEI"
    thompson_sampling: bool = True
    epsilon: float = 1e-2

    def validate(self) -> None:
        """Validate combinations of options before any expensive work starts."""
        if self.mode not in {"unconstrained", "preference", "constrained"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.num_ts_samples <= 0:
            raise ValueError("num_ts_samples must be positive.")
        if self.mode == "preference":
            if self.preference_region is None:
                raise ValueError("preference mode requires preference_region.")
        elif self.preference_region is not None:
            raise ValueError("preference_region should only be provided in preference mode.")
        if self.preference_region is not None and len(self.preference_region) != 2:
            raise ValueError("preference_region must contain exactly two points: lower and upper.")
