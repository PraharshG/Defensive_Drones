from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from defensive_drones.model import RunResult


def write_outputs(results: list[RunResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_per_run_csv(results, out_dir / "per_run_results.csv")
    summary_rows = summarize_results(results)
    write_summary_csv(summary_rows, out_dir / "summary.csv")
    write_graphs(results, out_dir)


def write_per_run_csv(results: list[RunResult], path: Path) -> None:
    fieldnames = [
        "run_id",
        "scenario_seed",
        "strategy",
        "defender_count",
        "attacker_count",
        "attacker_bucket",
        "kills",
        "breaches",
        "success",
        "completion_time_s",
        "kill_ratio",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "run_id": result.run_id,
                    "scenario_seed": result.scenario_seed,
                    "strategy": result.strategy,
                    "defender_count": result.defender_count,
                    "attacker_count": result.attacker_count,
                    "attacker_bucket": result.attacker_bucket,
                    "kills": result.kills,
                    "breaches": result.breaches,
                    "success": int(result.success),
                    "completion_time_s": f"{result.completion_time_s:.2f}",
                    "kill_ratio": f"{result.kill_ratio:.4f}",
                }
            )


def summarize_results(results: list[RunResult]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str], list[RunResult]] = defaultdict(list)
    for result in results:
        groups[
            (result.strategy, result.defender_count, result.attacker_bucket)
        ].append(result)

    rows: list[dict[str, object]] = []
    for (strategy, defender_count, attacker_bucket), grouped in sorted(groups.items()):
        success_rate = sum(result.success for result in grouped) / len(grouped)
        rows.append(
            {
                "strategy": strategy,
                "defender_count": defender_count,
                "attacker_bucket": attacker_bucket,
                "runs": len(grouped),
                "success_rate": round(success_rate, 4),
                "avg_kills": round(statistics.fmean(r.kills for r in grouped), 3),
                "avg_breaches": round(statistics.fmean(r.breaches for r in grouped), 3),
                "median_completion_time_s": round(
                    statistics.median(r.completion_time_s for r in grouped), 3
                ),
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "strategy",
        "defender_count",
        "attacker_bucket",
        "runs",
        "success_rate",
        "avg_kills",
        "avg_breaches",
        "median_completion_time_s",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_graphs(results: list[RunResult], out_dir: Path) -> None:
    _plot_success_by_strategy(results, out_dir / "success_rate_by_strategy.png")
    _plot_success_by_defender_count(results, out_dir / "success_rate_by_defender_count.png")
    _plot_success_by_attacker_bucket(results, out_dir / "success_rate_by_attacker_count.png")
    _plot_kill_ratio_distribution(results, out_dir / "kill_ratio_distribution.png")
    _plot_completion_time_distribution(results, out_dir / "completion_time_distribution.png")


def _strategies(results: list[RunResult]) -> list[str]:
    return sorted({result.strategy for result in results})


def _plot_success_by_strategy(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    rates = [
        _success_rate([result for result in results if result.strategy == strategy])
        for strategy in strategies
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(strategies, rates, color=["#3b82f6", "#10b981", "#f59e0b"][: len(strategies)])
    plt.ylabel("Success rate")
    plt.ylim(0.0, 1.0)
    plt.title("Success rate by strategy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_success_by_defender_count(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    defender_counts = sorted({result.defender_count for result in results})
    plt.figure(figsize=(8, 5))
    for strategy in strategies:
        rates = [
            _success_rate(
                [
                    result
                    for result in results
                    if result.strategy == strategy and result.defender_count == count
                ]
            )
            for count in defender_counts
        ]
        plt.plot(defender_counts, rates, marker="o", label=strategy)
    plt.xlabel("Defender count")
    plt.ylabel("Success rate")
    plt.ylim(0.0, 1.0)
    plt.title("Success rate by defender count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_success_by_attacker_bucket(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    buckets = ["5-10", "11-15", "16-20"]
    plt.figure(figsize=(8, 5))
    for strategy in strategies:
        rates = [
            _success_rate(
                [
                    result
                    for result in results
                    if result.strategy == strategy and result.attacker_bucket == bucket
                ]
            )
            for bucket in buckets
        ]
        plt.plot(buckets, rates, marker="o", label=strategy)
    plt.xlabel("Attacker count bucket")
    plt.ylabel("Success rate")
    plt.ylim(0.0, 1.0)
    plt.title("Success rate by attacker count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_kill_ratio_distribution(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    values = [
        [result.kill_ratio for result in results if result.strategy == strategy]
        for strategy in strategies
    ]
    plt.figure(figsize=(8, 5))
    plt.boxplot(values, tick_labels=strategies)
    plt.ylabel("Kill ratio")
    plt.ylim(0.0, 1.05)
    plt.title("Kill ratio distribution by strategy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_completion_time_distribution(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    values = [
        [
            result.completion_time_s
            for result in results
            if result.strategy == strategy and result.success
        ]
        for strategy in strategies
    ]
    non_empty_values = [value if value else [0.0] for value in values]
    plt.figure(figsize=(8, 5))
    plt.boxplot(non_empty_values, tick_labels=strategies)
    plt.ylabel("Completion time, successful runs (s)")
    plt.title("Completion time distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _success_rate(results: list[RunResult]) -> float:
    if not results:
        return 0.0
    return sum(result.success for result in results) / len(results)
