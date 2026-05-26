import argparse
import json
from pathlib import Path

from moo_constraints.config import MethodConfig
from moo_constraints.runner import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run the clean method-only MOBO export.")
    parser.add_argument("--problem", required=True, type=str)
    parser.add_argument("--mode", required=True, choices=["preference", "constrained"])
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--noise", default=0.0, type=float)
    parser.add_argument("--num-ts-samples", default=1, type=int)
    parser.add_argument("--output-dir", default="outputs", type=Path)
    parser.add_argument("--ref-point", nargs="+", type=float, default=None)
    parser.add_argument("--pref-lower", nargs="+", type=float, default=None)
    parser.add_argument("--pref-upper", nargs="+", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    preference_region = None
    if args.pref_lower is not None or args.pref_upper is not None:
        if args.pref_lower is None or args.pref_upper is None:
            raise ValueError("Both --pref-lower and --pref-upper must be provided together.")
        preference_region = [args.pref_lower, args.pref_upper]

    config = MethodConfig(
        problem=args.problem,
        mode=args.mode,
        steps=args.steps,
        seed=args.seed,
        noise=args.noise,
        num_ts_samples=args.num_ts_samples,
        output_dir=args.output_dir,
        ref_point=args.ref_point,
        preference_region=preference_region,
    )
    result = run_experiment(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
