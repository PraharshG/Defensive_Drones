from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import numpy as np

from defensive_drones.engine import simulate_scenario
from defensive_drones.model import (
    RunResult,
    Scenario,
    SimulationConfig,
    initial_defender_count_for_scenario,
)
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
    jobs: int = 1,
) -> list[RunResult]:
    config = config or SimulationConfig()
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    master_rng = np.random.default_rng(seed)
    scenario_seeds = [
        int(master_rng.integers(0, np.iinfo(np.uint32).max)) for _ in range(runs)
    ]
    defender_counts = config.defender_counts
    results: list[RunResult] = []
    status_log_path = log_path or out_dir / "run_status.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunStatusLogger(status_log_path) as logger:
        logger.started(runs, seed, strategies, defender_counts, jobs)
        if jobs == 1:
            for run_id, scenario_seed in enumerate(scenario_seeds):
                for defender_count in defender_counts:
                    cell_result = _run_scenario_cell(
                        run_id, scenario_seed, defender_count, config, strategies
                    )
                    logger.scenario_started_counts(
                        run_id,
                        scenario_seed,
                        cell_result.defender_count,
                        cell_result.total_defender_count,
                        cell_result.attacker_count,
                        cell_result.wave_count,
                    )
                    results.extend(cell_result.results)
                    for result in cell_result.results:
                        logger.strategy_completed(result)
                    logger.scenario_completed(run_id, cell_result.results)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = []
                for run_id, scenario_seed in enumerate(scenario_seeds):
                    for defender_count in defender_counts:
                        logger.scenario_cell_queued(
                            run_id, scenario_seed, defender_count
                        )
                        futures.append(
                            executor.submit(
                                _run_scenario_cell,
                                run_id,
                                scenario_seed,
                                defender_count,
                                config,
                                strategies,
                            )
                        )
                for future in as_completed(futures):
                    cell_result = future.result()
                    results.extend(cell_result.results)
                    for result in cell_result.results:
                        logger.strategy_completed(result)
                    logger.scenario_completed(
                        cell_result.run_id, cell_result.results
                    )
        logger.finished(results, out_dir)

    results.sort(
        key=lambda result: (
            result.run_id,
            result.scenario_seed,
            result.defender_count,
            strategies.index(result.strategy),
        )
    )
    write_outputs(results, out_dir)
    return results


@dataclass(frozen=True)
class ScenarioCellResult:
    run_id: int
    scenario_seed: int
    defender_count: int
    total_defender_count: int
    attacker_count: int
    wave_count: int
    results: list[RunResult]


def _run_scenario_cell(
    run_id: int,
    scenario_seed: int,
    defender_count: int,
    config: SimulationConfig,
    strategies: tuple[str, ...],
) -> ScenarioCellResult:
    scenario = generate_scenario(scenario_seed, config, defender_count=defender_count)
    scenario_results = [
        simulate_scenario(
            scenario,
            config,
            strategy=strategy,
            run_id=run_id,
            observation_seed=scenario_seed,
        )
        for strategy in strategies
    ]
    return ScenarioCellResult(
        run_id=run_id,
        scenario_seed=scenario_seed,
        defender_count=initial_defender_count_for_scenario(scenario),
        total_defender_count=len(scenario.defenders),
        attacker_count=len(scenario.attackers),
        wave_count=len({attacker.wave_id for attacker in scenario.attackers}),
        results=scenario_results,
    )


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
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel scenario cells to run. Use 1 for serial execution.",
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
        jobs=args.jobs,
    )
    _print_console_summary(results, args.out, args.log or args.out / "run_status.log")
    return 0


def _print_console_summary(
    results: list[RunResult], out_dir: Path, log_path: Path
) -> None:
    print(f"Wrote {len(results)} strategy runs to {out_dir}")
    print(f"Run status log: {log_path}")
    print(
        "strategy, defenders, attackers, runs, success_rate, avg_kills, "
        "avg_breaches, avg_breach_rate, avg_contention_rate"
    )
    for row in summarize_results(results):
        print(
            f"{row['strategy']}, {row['defender_count']}, {row['attacker_bucket']}, "
            f"{row['runs']}, {row['success_rate']}, {row['avg_kills']}, "
            f"{row['avg_breaches']}, {row['avg_breach_rate']}, "
            f"{row['avg_contention_rate']}"
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

    def started(
        self,
        runs: int,
        seed: int,
        strategies: tuple[str, ...],
        defender_counts: tuple[int, ...],
        jobs: int,
    ) -> None:
        self._write(
            "START "
            f"runs={runs} seed={seed} strategy_count={len(strategies)} "
            f"strategies={','.join(strategies)} "
            f"defender_counts={','.join(str(count) for count in defender_counts)} "
            f"scenario_cells={runs * len(defender_counts)} jobs={jobs}"
        )

    def scenario_started(
        self, run_id: int, scenario_seed: int, scenario: Scenario
    ) -> None:
        self.scenario_started_counts(
            run_id,
            scenario_seed,
            initial_defender_count_for_scenario(scenario),
            len(scenario.defenders),
            len(scenario.attackers),
            len({attacker.wave_id for attacker in scenario.attackers}),
        )

    def scenario_started_counts(
        self,
        run_id: int,
        scenario_seed: int,
        defender_count: int,
        total_defender_count: int,
        attacker_count: int,
        wave_count: int,
    ) -> None:
        self._write(
            "SCENARIO_START "
            f"run_id={run_id} scenario_seed={scenario_seed} "
            f"defenders={defender_count} total_defenders={total_defender_count} "
            f"attackers={attacker_count} "
            f"waves={wave_count}"
        )

    def scenario_cell_queued(
        self, run_id: int, scenario_seed: int, defender_count: int
    ) -> None:
        self._write(
            "SCENARIO_START "
            f"run_id={run_id} scenario_seed={scenario_seed} "
            f"defenders={defender_count} attackers=pending"
        )

    def strategy_completed(self, result: RunResult) -> None:
        self._write(
            "STRATEGY_DONE "
            f"run_id={result.run_id} strategy={result.strategy} "
            f"defenders={result.defender_count} "
            f"total_defenders={result.total_defender_count} "
            f"attackers={result.attacker_count} "
            f"waves={result.wave_count} "
            f"kills={result.kills} breaches={result.breaches} "
            f"breach_rate={result.breach_rate:.4f} "
            f"contention_rate={result.contention_rate:.4f} "
            f"max_target_contention={result.max_simultaneous_defenders_on_target} "
            f"success={int(result.success)} "
            f"completion_time_s={result.completion_time_s:.2f} "
            f"first_breach_time_s={_format_optional_seconds(result.first_breach_time_s)}"
        )

    def scenario_completed(
        self, run_id: int, scenario_results: list[RunResult]
    ) -> None:
        successes = sum(result.success for result in scenario_results)
        best_kills = max((result.kills for result in scenario_results), default=0)
        defender_count = (
            scenario_results[0].defender_count if scenario_results else "unknown"
        )
        total_defender_count = (
            scenario_results[0].total_defender_count
            if scenario_results
            else "unknown"
        )
        self._write(
            "SCENARIO_DONE "
            f"run_id={run_id} strategy_runs={len(scenario_results)} "
            f"defenders={defender_count} total_defenders={total_defender_count} "
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


def _format_optional_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
