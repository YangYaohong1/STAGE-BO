from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MethodConfig:
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

    # The export keeps only the method variants that are actively used.
    constraint_strategy: str = "maxmin"
    optimize_method: str = "posterior"
    inner_acquisition: str = "cEI"
    thompson_sampling: bool = True
    epsilon: float = 1e-2

    def validate(self) -> None:
        if self.mode not in {"preference", "constrained"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.constraint_strategy != "maxmin":
            raise ValueError("This clean export only supports constraint_strategy='maxmin'.")
        if self.optimize_method != "posterior":
            raise ValueError("This clean export only supports optimize_method='posterior'.")
        if self.inner_acquisition != "cEI":
            raise ValueError("This clean export only supports inner_acquisition='cEI'.")
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.num_ts_samples <= 0:
            raise ValueError("num_ts_samples must be positive.")
        if self.preference_region is not None:
            if len(self.preference_region) != 2:
                raise ValueError("preference_region must contain exactly two points: lower and upper.")
