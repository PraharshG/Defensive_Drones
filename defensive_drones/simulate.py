from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import numpy as np

from defensive_drones.engine import simulate_scenario
from defensive_drones.model import RunResult, Scenario, SimulationConfig
from defensive_drones.reporting import summarize_results, write_outputs
from defensive_drones.scenario import generate_scenario
from defensive_drones.strategies import STRATEGIES


def run_monte_carlo(
    runs: int,
    seed: int,
    out_dir: Path,
    config: SimulationConfig | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
    log_path: Path | None = None,
) -> list[RunResult]:
    config = config or SimulationConfig()
    master_rng = np.random.default_rng(seed)
    results: list[RunResult] = []
    status_log_path = log_path or out_dir / "run_status.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunStatusLogger(status_log_path) as logger:
        logger.started(runs, seed, strategies)
        for run_id in range(runs):
            scenario_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
            scenario = generate_scenario(scenario_seed, config)
            logger.scenario_started(run_id, scenario_seed, scenario)

            scenario_results: list[RunResult] = []
            for strategy in strategies:
                observation_seed = scenario_seed
                result = simulate_scenario(
                    scenario,
                    config,
                    strategy=strategy,
                    run_id=run_id,
                    observation_seed=observation_seed,
                )
                results.append(result)
                scenario_results.append(result)
                logger.strategy_completed(result)

            logger.scenario_completed(run_id, scenario_results)
        logger.finished(results, out_dir)

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
        "--log",
        type=Path,
        default=None,
        help="Run status log path. Defaults to <out>/run_status.log.",
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
        log_path=args.log,
    )
    _print_console_summary(results, args.out, args.log or args.out / "run_status.log")
    return 0


def _print_console_summary(
    results: list[RunResult], out_dir: Path, log_path: Path
) -> None:
    print(f"Wrote {len(results)} strategy runs to {out_dir}")
    print(f"Run status log: {log_path}")
    print("strategy, defenders, attackers, runs, success_rate, avg_kills, avg_breaches")
    for row in summarize_results(results):
        print(
            f"{row['strategy']}, {row['defender_count']}, {row['attacker_bucket']}, "
            f"{row['runs']}, {row['success_rate']}, {row['avg_kills']}, "
            f"{row['avg_breaches']}"
        )


class RunStatusLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: TextIO | None = None

    def __enter__(self) -> "RunStatusLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            self._write(f"ERROR type={exc_type.__name__} message={exc}")
        if self._handle is not None:
            self._handle.close()

    def started(self, runs: int, seed: int, strategies: tuple[str, ...]) -> None:
        self._write(
            "START "
            f"runs={runs} seed={seed} strategy_count={len(strategies)} "
            f"strategies={','.join(strategies)}"
        )

    def scenario_started(
        self, run_id: int, scenario_seed: int, scenario: Scenario
    ) -> None:
        self._write(
            "SCENARIO_START "
            f"run_id={run_id} scenario_seed={scenario_seed} "
            f"defenders={len(scenario.defenders)} attackers={len(scenario.attackers)}"
        )

    def strategy_completed(self, result: RunResult) -> None:
        self._write(
            "STRATEGY_DONE "
            f"run_id={result.run_id} strategy={result.strategy} "
            f"defenders={result.defender_count} attackers={result.attacker_count} "
            f"kills={result.kills} breaches={result.breaches} "
            f"success={int(result.success)} "
            f"completion_time_s={result.completion_time_s:.2f}"
        )

    def scenario_completed(
        self, run_id: int, scenario_results: list[RunResult]
    ) -> None:
        successes = sum(result.success for result in scenario_results)
        best_kills = max((result.kills for result in scenario_results), default=0)
        self._write(
            "SCENARIO_DONE "
            f"run_id={run_id} strategy_runs={len(scenario_results)} "
            f"successful_strategies={successes} best_kills={best_kills}"
        )

    def finished(self, results: list[RunResult], out_dir: Path) -> None:
        successes = sum(result.success for result in results)
        self._write(
            "FINISH "
            f"strategy_runs={len(results)} successes={successes} output_dir={out_dir}"
        )

    def _write(self, message: str) -> None:
        if self._handle is None:
            raise RuntimeError("RunStatusLogger is not open")
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._handle.write(f"{timestamp} {message}\n")
        self._handle.flush()


if __name__ == "__main__":
    raise SystemExit(main())
