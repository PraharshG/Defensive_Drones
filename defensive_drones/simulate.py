from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from defensive_drones.engine import simulate_scenario
from defensive_drones.model import RunResult, SimulationConfig
from defensive_drones.reporting import summarize_results, write_outputs
from defensive_drones.scenario import generate_scenario
from defensive_drones.strategies import STRATEGIES


def run_monte_carlo(
    runs: int,
    seed: int,
    out_dir: Path,
    config: SimulationConfig | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
) -> list[RunResult]:
    config = config or SimulationConfig()
    master_rng = np.random.default_rng(seed)
    results: list[RunResult] = []

    for run_id in range(runs):
        scenario_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        scenario = generate_scenario(scenario_seed, config)
        for strategy in strategies:
            observation_seed = scenario_seed
            results.append(
                simulate_scenario(
                    scenario,
                    config,
                    strategy=strategy,
                    run_id=run_id,
                    observation_seed=observation_seed,
                )
            )

    write_outputs(results, out_dir)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run defensive drone Monte Carlo simulations."
    )
    parser.add_argument("--runs", type=int, default=1000, help="Number of scenarios.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs"), help="Output directory."
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
        help="Strategies to evaluate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = run_monte_carlo(
        runs=args.runs,
        seed=args.seed,
        out_dir=args.out,
        strategies=tuple(args.strategies),
    )
    _print_console_summary(results, args.out)
    return 0


def _print_console_summary(results: list[RunResult], out_dir: Path) -> None:
    print(f"Wrote {len(results)} strategy runs to {out_dir}")
    print("strategy, defenders, attackers, runs, success_rate, avg_kills, avg_breaches")
    for row in summarize_results(results):
        print(
            f"{row['strategy']}, {row['defender_count']}, {row['attacker_bucket']}, "
            f"{row['runs']}, {row['success_rate']}, {row['avg_kills']}, "
            f"{row['avg_breaches']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
