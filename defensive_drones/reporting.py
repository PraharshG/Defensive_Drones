from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from defensive_drones.model import RunResult, WaveResult, attacker_bucket_sort_key


PLOT_COLORS = [
    "#0072B2",
    "#56B4E9",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#D55E00",
    "#4B5563",
    "#9333EA",
    "#0F766E",
]


def write_outputs(results: list[RunResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_per_run_csv(results, out_dir / "per_run_results.csv")
    write_wave_summary_csv(results, out_dir / "wave_summary.csv")
    summary_rows = summarize_results(results)
    write_summary_csv(summary_rows, out_dir / "summary.csv")
    write_success_rate_matrix_csv(results, out_dir / "success_rate_matrix.csv")
    write_breach_rate_matrix_csv(results, out_dir / "breach_rate_matrix.csv")
    write_graphs(results, out_dir)


def write_per_run_csv(results: list[RunResult], path: Path) -> None:
    fieldnames = [
        "run_id",
        "scenario_seed",
        "strategy",
        "defender_count",
        "attacker_count",
        "wave_count",
        "attacker_bucket",
        "kills",
        "breaches",
        "success",
        "completion_time_s",
        "kill_ratio",
        "breach_rate",
        "first_breach_time_s",
        "duplicate_target_assignments",
        "total_assignments",
        "contention_rate",
        "max_simultaneous_defenders_on_target",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "run_id": result.run_id,
                    "scenario_seed": result.scenario_seed,
                    "strategy": result.strategy,
                    "defender_count": result.defender_count,
                    "attacker_count": result.attacker_count,
                    "wave_count": result.wave_count,
                    "attacker_bucket": result.attacker_bucket,
                    "kills": result.kills,
                    "breaches": result.breaches,
                    "success": int(result.success),
                    "completion_time_s": f"{result.completion_time_s:.2f}",
                    "kill_ratio": f"{result.kill_ratio:.4f}",
                    "breach_rate": f"{result.breach_rate:.4f}",
                    "first_breach_time_s": _format_optional_seconds(
                        result.first_breach_time_s
                    ),
                    "duplicate_target_assignments": result.duplicate_target_assignments,
                    "total_assignments": result.total_assignments,
                    "contention_rate": f"{result.contention_rate:.4f}",
                    "max_simultaneous_defenders_on_target": (
                        result.max_simultaneous_defenders_on_target
                    ),
                }
            )


def write_wave_summary_csv(results: list[RunResult], path: Path) -> None:
    fieldnames = [
        "run_id",
        "scenario_seed",
        "strategy",
        "defender_count",
        "wave_id",
        "spawn_time_s",
        "attacker_count",
        "kills",
        "breaches",
        "success",
        "completion_time_s",
        "kill_ratio",
        "breach_rate",
        "first_breach_time_s",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for wave in _wave_results(results):
            writer.writerow(
                {
                    "run_id": wave.run_id,
                    "scenario_seed": wave.scenario_seed,
                    "strategy": wave.strategy,
                    "defender_count": wave.defender_count,
                    "wave_id": wave.wave_id,
                    "spawn_time_s": f"{wave.spawn_time_s:.2f}",
                    "attacker_count": wave.attacker_count,
                    "kills": wave.kills,
                    "breaches": wave.breaches,
                    "success": int(wave.success),
                    "completion_time_s": f"{wave.completion_time_s:.2f}",
                    "kill_ratio": f"{wave.kill_ratio:.4f}",
                    "breach_rate": f"{wave.breach_rate:.4f}",
                    "first_breach_time_s": _format_optional_seconds(
                        wave.first_breach_time_s
                    ),
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
                "avg_breach_rate": round(
                    statistics.fmean(r.breach_rate for r in grouped), 4
                ),
                "median_breaches": round(
                    statistics.median(r.breaches for r in grouped), 3
                ),
                "max_breaches": max(r.breaches for r in grouped),
                "median_first_breach_time_s": _median_optional_seconds(
                    [r.first_breach_time_s for r in grouped]
                ),
                "avg_contention_rate": round(
                    statistics.fmean(r.contention_rate for r in grouped), 4
                ),
                "avg_duplicate_target_assignments": round(
                    statistics.fmean(
                        r.duplicate_target_assignments for r in grouped
                    ),
                    3,
                ),
                "max_simultaneous_defenders_on_target": max(
                    r.max_simultaneous_defenders_on_target for r in grouped
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
        "avg_breach_rate",
        "median_breaches",
        "max_breaches",
        "median_first_breach_time_s",
        "avg_contention_rate",
        "avg_duplicate_target_assignments",
        "max_simultaneous_defenders_on_target",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_success_rate_matrix_csv(results: list[RunResult], path: Path) -> None:
    strategies = sorted({result.strategy for result in results})
    defender_counts = sorted({result.defender_count for result in results})
    attacker_buckets = _attacker_buckets(results)
    fieldnames = ["strategy", "defense_drones"] + [
        f"attack_{bucket.replace('-', '_')}" for bucket in attacker_buckets
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for strategy in strategies:
            for defender_count in defender_counts:
                row: dict[str, object] = {
                    "strategy": strategy,
                    "defense_drones": defender_count,
                }
                for attacker_bucket in attacker_buckets:
                    subset = [
                        result
                        for result in results
                        if result.strategy == strategy
                        and result.defender_count == defender_count
                        and result.attacker_bucket == attacker_bucket
                    ]
                    column = f"attack_{attacker_bucket.replace('-', '_')}"
                    if subset:
                        row[column] = round(
                            sum(result.success for result in subset) / len(subset), 4
                        )
                    else:
                        row[column] = ""
                writer.writerow(row)


def write_breach_rate_matrix_csv(results: list[RunResult], path: Path) -> None:
    strategies = sorted({result.strategy for result in results})
    defender_counts = sorted({result.defender_count for result in results})
    attacker_buckets = _attacker_buckets(results)
    fieldnames = ["strategy", "defense_drones"] + [
        f"attack_{bucket.replace('-', '_')}" for bucket in attacker_buckets
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for strategy in strategies:
            for defender_count in defender_counts:
                row: dict[str, object] = {
                    "strategy": strategy,
                    "defense_drones": defender_count,
                }
                for attacker_bucket in attacker_buckets:
                    subset = [
                        result
                        for result in results
                        if result.strategy == strategy
                        and result.defender_count == defender_count
                        and result.attacker_bucket == attacker_bucket
                    ]
                    column = f"attack_{attacker_bucket.replace('-', '_')}"
                    if subset:
                        row[column] = round(
                            statistics.fmean(result.breach_rate for result in subset),
                            4,
                        )
                    else:
                        row[column] = ""
                writer.writerow(row)


def write_graphs(results: list[RunResult], out_dir: Path) -> None:
    _plot_success_by_strategy(results, out_dir / "success_rate_by_strategy.png")
    _plot_success_by_defender_count(results, out_dir / "success_rate_by_defender_count.png")
    _plot_success_by_attacker_bucket(results, out_dir / "success_rate_by_attacker_count.png")
    _plot_kill_ratio_distribution(results, out_dir / "kill_ratio_distribution.png")
    _plot_completion_time_distribution(results, out_dir / "completion_time_distribution.png")
    _plot_breach_rate_by_defender_count(results, out_dir / "breach_rate_by_defender_count.png")
    _plot_avg_breaches_by_defender_count(results, out_dir / "avg_breaches_by_defender_count.png")
    _plot_breach_rate_distribution(results, out_dir / "breach_rate_distribution.png")
    _plot_first_breach_time_distribution(results, out_dir / "first_breach_time_distribution.png")
    _plot_breach_rate_by_wave(results, out_dir / "breach_rate_by_wave.png")
    _plot_kill_ratio_by_wave(results, out_dir / "kill_ratio_by_wave.png")
    _plot_contention_rate_by_strategy(results, out_dir / "contention_rate_by_strategy.png")
    _plot_max_contention_distribution(results, out_dir / "max_contention_distribution.png")


def _strategies(results: list[RunResult]) -> list[str]:
    return sorted({result.strategy for result in results})


def _plot_success_by_strategy(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    rates = [
        _success_rate([result for result in results if result.strategy == strategy])
        for strategy in strategies
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(
        strategies,
        rates,
        color=[PLOT_COLORS[index % len(PLOT_COLORS)] for index in range(len(strategies))],
    )
    plt.ylabel("Success rate")
    plt.ylim(0.0, 1.0)
    plt.title("Success rate by strategy")
    plt.xticks(rotation=30, ha="right")
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
    buckets = _attacker_buckets(results)
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
    _boxplot(values, strategies)
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
    _boxplot(non_empty_values, strategies)
    plt.ylabel("Completion time, successful runs (s)")
    plt.title("Completion time distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_breach_rate_by_defender_count(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    defender_counts = sorted({result.defender_count for result in results})
    plt.figure(figsize=(8, 5))
    for strategy in strategies:
        rates = [
            _mean(
                [
                    result.breach_rate
                    for result in results
                    if result.strategy == strategy and result.defender_count == count
                ]
            )
            for count in defender_counts
        ]
        plt.plot(defender_counts, rates, marker="o", label=strategy)
    plt.xlabel("Defender count")
    plt.ylabel("Mean pass-through rate")
    plt.ylim(0.0, 1.0)
    plt.title("Pass-through rate by defender count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_avg_breaches_by_defender_count(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    defender_counts = sorted({result.defender_count for result in results})
    plt.figure(figsize=(8, 5))
    for strategy in strategies:
        breaches = [
            _mean(
                [
                    float(result.breaches)
                    for result in results
                    if result.strategy == strategy and result.defender_count == count
                ]
            )
            for count in defender_counts
        ]
        plt.plot(defender_counts, breaches, marker="o", label=strategy)
    plt.xlabel("Defender count")
    plt.ylabel("Mean attackers passing through")
    plt.title("Attackers passing through by defender count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_breach_rate_distribution(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    values = [
        [result.breach_rate for result in results if result.strategy == strategy]
        for strategy in strategies
    ]
    plt.figure(figsize=(8, 5))
    _boxplot(values, strategies)
    plt.ylabel("Pass-through rate")
    plt.ylim(0.0, 1.05)
    plt.title("Pass-through rate distribution by strategy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_first_breach_time_distribution(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    values = [
        [
            result.first_breach_time_s
            for result in results
            if result.strategy == strategy and result.first_breach_time_s is not None
        ]
        for strategy in strategies
    ]
    non_empty_values = [value if value else [0.0] for value in values]
    plt.figure(figsize=(8, 5))
    _boxplot(non_empty_values, strategies)
    plt.ylabel("First pass-through time (s)")
    plt.title("First pass-through time distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_breach_rate_by_wave(results: list[RunResult], path: Path) -> None:
    waves = _wave_results(results)
    strategies = _strategies(results)
    wave_ids = sorted({wave.wave_id for wave in waves})
    plt.figure(figsize=(8, 5))
    for strategy in strategies:
        rates = [
            _mean(
                [
                    wave.breach_rate
                    for wave in waves
                    if wave.strategy == strategy and wave.wave_id == wave_id
                ]
            )
            for wave_id in wave_ids
        ]
        plt.plot(wave_ids, rates, marker="o", label=strategy)
    plt.xlabel("Wave")
    plt.ylabel("Mean pass-through rate")
    plt.ylim(0.0, 1.0)
    plt.title("Pass-through rate by attacker wave")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_kill_ratio_by_wave(results: list[RunResult], path: Path) -> None:
    waves = _wave_results(results)
    strategies = _strategies(results)
    wave_ids = sorted({wave.wave_id for wave in waves})
    plt.figure(figsize=(8, 5))
    for strategy in strategies:
        rates = [
            _mean(
                [
                    wave.kill_ratio
                    for wave in waves
                    if wave.strategy == strategy and wave.wave_id == wave_id
                ]
            )
            for wave_id in wave_ids
        ]
        plt.plot(wave_ids, rates, marker="o", label=strategy)
    plt.xlabel("Wave")
    plt.ylabel("Mean kill ratio")
    plt.ylim(0.0, 1.0)
    plt.title("Kill ratio by attacker wave")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_contention_rate_by_strategy(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    rates = [
        _mean(
            [result.contention_rate for result in results if result.strategy == strategy]
        )
        for strategy in strategies
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(
        strategies,
        rates,
        color=[PLOT_COLORS[index % len(PLOT_COLORS)] for index in range(len(strategies))],
    )
    plt.ylabel("Mean duplicate assignment rate")
    plt.ylim(0.0, 1.0)
    plt.title("Target contention by strategy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_max_contention_distribution(results: list[RunResult], path: Path) -> None:
    strategies = _strategies(results)
    values = [
        [
            float(result.max_simultaneous_defenders_on_target)
            for result in results
            if result.strategy == strategy
        ]
        for strategy in strategies
    ]
    plt.figure(figsize=(8, 5))
    _boxplot(values, strategies)
    plt.ylabel("Max defenders assigned to one attacker")
    plt.title("Peak target contention distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _success_rate(results: list[RunResult]) -> float:
    if not results:
        return 0.0
    return sum(result.success for result in results) / len(results)


def _attacker_buckets(results: list[RunResult]) -> list[str]:
    return sorted(
        {result.attacker_bucket for result in results},
        key=attacker_bucket_sort_key,
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.fmean(values)


def _wave_results(results: list[RunResult]) -> list[WaveResult]:
    return [wave for result in results for wave in result.wave_results]


def _median_optional_seconds(values: list[float | None]) -> object:
    present = [value for value in values if value is not None]
    if not present:
        return ""
    return round(statistics.median(present), 3)


def _format_optional_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}"


def _boxplot(values: list[list[float]], labels: list[str]) -> None:
    try:
        plt.boxplot(values, tick_labels=labels)
    except TypeError:
        plt.boxplot(values, labels=labels)
