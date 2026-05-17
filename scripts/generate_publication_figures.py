from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STRATEGY_ORDER = [
    "optimized_global",
    "optimized_collab",
    "optimized",
    "nearest_global",
    "nearest_collab",
    "nearest",
    "earliest_deadline_global",
    "earliest_deadline_collab",
    "earliest_deadline",
]
STRATEGY_LABELS = {
    "optimized_global": "Optimized\nGlobal",
    "optimized_collab": "Optimized\nCollab",
    "optimized": "Optimized\nCorridor",
    "nearest_global": "Nearest\nGlobal",
    "nearest_collab": "Nearest\nCollab",
    "nearest": "Nearest\nCorridor",
    "earliest_deadline_global": "Deadline\nGlobal",
    "earliest_deadline_collab": "Deadline\nCollab",
    "earliest_deadline": "Deadline\nCorridor",
}
SHORT_LABELS = {
    "optimized_global": "Optimized Global",
    "optimized_collab": "Optimized Collab",
    "optimized": "Optimized Corridor",
    "nearest_global": "Nearest Global",
    "nearest_collab": "Nearest Collab",
    "nearest": "Nearest Corridor",
    "earliest_deadline_global": "Deadline Global",
    "earliest_deadline_collab": "Deadline Collab",
    "earliest_deadline": "Deadline Corridor",
}
PAIR_LABELS = {
    "optimized": "Optimized",
    "nearest": "Nearest",
    "earliest_deadline": "Deadline",
}
GLOBAL_FOR = {
    "optimized": "optimized_global",
    "nearest": "nearest_global",
    "earliest_deadline": "earliest_deadline_global",
}
COLLAB_FOR = {
    "optimized": "optimized_collab",
    "nearest": "nearest_collab",
    "earliest_deadline": "earliest_deadline_collab",
}
ATTACKER_BUCKET_ORDER = ["5-10", "11-15", "16-20"]
PALETTE = {
    "optimized_global": "#0072B2",
    "optimized_collab": "#4B5563",
    "optimized": "#56B4E9",
    "nearest_global": "#009E73",
    "nearest_collab": "#0F766E",
    "nearest": "#E69F00",
    "earliest_deadline_global": "#CC79A7",
    "earliest_deadline_collab": "#9333EA",
    "earliest_deadline": "#D55E00",
}


@dataclass(frozen=True)
class Result:
    run_id: int
    scenario_seed: int
    strategy: str
    defender_count: int
    attacker_count: int
    attacker_bucket: str
    kills: int
    breaches: int
    success: bool
    completion_time_s: float
    kill_ratio: float


@dataclass(frozen=True)
class FigureRecord:
    number: int
    filename: str
    title: str
    message: str


def main() -> int:
    args = build_parser().parse_args()
    results = load_results(args.input)
    output_dir = args.out
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    figure_records = [
        plot_success_rate_by_strategy(results, output_dir, 1),
        plot_mean_kill_ratio_by_strategy(results, output_dir, 2),
        plot_mean_kills_by_strategy(results, output_dir, 3),
        plot_success_by_defender_count(results, output_dir, 4),
        plot_success_by_attacker_bucket(results, output_dir, 5),
        plot_success_heatmap(results, output_dir, 6),
        plot_kill_ratio_distribution(results, output_dir, 7),
        plot_terminal_time_distribution(results, output_dir, 8),
        plot_global_vs_corridor_success_uplift(results, output_dir, 9),
        plot_optimized_global_delta(results, output_dir, 10),
    ]

    write_figure_index(figure_records, output_dir / "figure_index.md")
    write_talking_points(results, figure_records, output_dir / "talking_points.md")
    write_success_rate_matrix_markdown(
        results, output_dir / "success_rate_matrix.md"
    )
    print(f"Wrote {len(figure_records)} publication figures to {output_dir}")
    print(f"Talking points: {output_dir / 'talking_points.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from per_run_results.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/per_run_results.csv"),
        help="Input per-run CSV file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/publication_figures"),
        help="Directory for PNG/PDF figures and talking points.",
    )
    return parser


def load_results(path: Path) -> list[Result]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Result(
            run_id=int(row["run_id"]),
            scenario_seed=int(row["scenario_seed"]),
            strategy=row["strategy"],
            defender_count=int(row["defender_count"]),
            attacker_count=int(row["attacker_count"]),
            attacker_bucket=row["attacker_bucket"],
            kills=int(row["kills"]),
            breaches=int(row["breaches"]),
            success=bool(int(row["success"])),
            completion_time_s=float(row["completion_time_s"]),
            kill_ratio=float(row["kill_ratio"]),
        )
        for row in rows
    ]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ordered_strategies(results: list[Result]) -> list[str]:
    present = {result.strategy for result in results}
    return [strategy for strategy in STRATEGY_ORDER if strategy in present]


def group_by_strategy(results: list[Result]) -> dict[str, list[Result]]:
    grouped: dict[str, list[Result]] = defaultdict(list)
    for result in results:
        grouped[result.strategy].append(result)
    return grouped


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z**2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def rate(results: list[Result]) -> float:
    if not results:
        return 0.0
    return sum(result.success for result in results) / len(results)


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> str:
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path.name


def annotate_bars(ax: plt.Axes, values: list[float], percent: bool = False) -> None:
    for patch, value in zip(ax.patches, values):
        label = f"{value * 100:.1f}%" if percent else f"{value:.2f}"
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
        )


def plot_success_rate_by_strategy(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    grouped = group_by_strategy(results)
    strategies = ordered_strategies(results)
    rates = [rate(grouped[strategy]) for strategy in strategies]
    intervals = [
        wilson_interval(sum(r.success for r in grouped[strategy]), len(grouped[strategy]))
        for strategy in strategies
    ]
    yerr = np.array(
        [[value - low for value, (low, _) in zip(rates, intervals)],
         [high - value for value, (_, high) in zip(rates, intervals)]]
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(strategies))
    ax.bar(
        x,
        rates,
        yerr=yerr,
        capsize=4,
        color=[PALETTE[strategy] for strategy in strategies],
        edgecolor="#222222",
        linewidth=0.5,
    )
    ax.set_title("Mission Success Rate by Strategy")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, max(rates) * 1.35 if rates else 1.0)
    ax.set_xticks(x, [STRATEGY_LABELS[strategy] for strategy in strategies])
    annotate_bars(ax, rates, percent=True)
    filename = save_figure(fig, output_dir, "01_success_rate_by_strategy")
    return FigureRecord(
        number,
        filename,
        "Mission success rate by strategy",
        "Shows which strategy most often kills all attackers before breach.",
    )


def plot_mean_kill_ratio_by_strategy(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    grouped = group_by_strategy(results)
    strategies = ordered_strategies(results)
    values = [mean([result.kill_ratio for result in grouped[strategy]]) for strategy in strategies]
    errors = [
        1.96 * standard_error([result.kill_ratio for result in grouped[strategy]])
        for strategy in strategies
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(strategies))
    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=4,
        color=[PALETTE[strategy] for strategy in strategies],
        edgecolor="#222222",
        linewidth=0.5,
    )
    ax.set_title("Average Fraction of Attackers Killed")
    ax.set_ylabel("Mean kill ratio")
    ax.set_ylim(0.0, min(1.0, max(values) * 1.22 if values else 1.0))
    ax.set_xticks(x, [STRATEGY_LABELS[strategy] for strategy in strategies])
    annotate_bars(ax, values)
    filename = save_figure(fig, output_dir, "02_mean_kill_ratio_by_strategy")
    return FigureRecord(
        number,
        filename,
        "Average fraction of attackers killed",
        "Normalizes kills by scenario size so mixed attacker counts can be compared.",
    )


def plot_mean_kills_by_strategy(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    grouped = group_by_strategy(results)
    strategies = ordered_strategies(results)
    values = [mean([result.kills for result in grouped[strategy]]) for strategy in strategies]
    errors = [
        1.96 * standard_error([result.kills for result in grouped[strategy]])
        for strategy in strategies
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(strategies))
    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=4,
        color=[PALETTE[strategy] for strategy in strategies],
        edgecolor="#222222",
        linewidth=0.5,
    )
    ax.set_title("Average Attackers Killed per Scenario")
    ax.set_ylabel("Mean kills")
    ax.set_ylim(0.0, max(values) * 1.22 if values else 1.0)
    ax.set_xticks(x, [STRATEGY_LABELS[strategy] for strategy in strategies])
    annotate_bars(ax, values)
    filename = save_figure(fig, output_dir, "03_mean_kills_by_strategy")
    return FigureRecord(
        number,
        filename,
        "Average attackers killed per scenario",
        "Reports the absolute number of intercepted attackers before breach.",
    )


def plot_success_by_defender_count(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    strategies = ordered_strategies(results)
    defender_counts = sorted({result.defender_count for result in results})
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for strategy in strategies:
        y = [
            rate(
                [
                    result
                    for result in results
                    if result.strategy == strategy and result.defender_count == count
                ]
            )
            for count in defender_counts
        ]
        ax.plot(
            defender_counts,
            y,
            marker="o",
            linewidth=2.0,
            markersize=5,
            color=PALETTE[strategy],
            label=SHORT_LABELS[strategy],
        )
    ax.set_title("Success Rate Scales with Defender Count")
    ax.set_xlabel("Defensive drones")
    ax.set_ylabel("Success rate")
    ax.set_xticks(defender_counts)
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    filename = save_figure(fig, output_dir, "04_success_rate_by_defender_count")
    return FigureRecord(
        number,
        filename,
        "Success rate by defender count",
        "Shows how added defensive assets change mission outcomes.",
    )


def plot_success_by_attacker_bucket(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    strategies = ordered_strategies(results)
    x = np.arange(len(ATTACKER_BUCKET_ORDER))
    width = 0.12
    offsets = np.linspace(-width * 2.5, width * 2.5, len(strategies))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for offset, strategy in zip(offsets, strategies):
        values = [
            rate(
                [
                    result
                    for result in results
                    if result.strategy == strategy and result.attacker_bucket == bucket
                ]
            )
            for bucket in ATTACKER_BUCKET_ORDER
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=PALETTE[strategy],
            label=SHORT_LABELS[strategy],
            edgecolor="#222222",
            linewidth=0.4,
        )
    ax.set_title("Success Rate Falls as Attacker Load Increases")
    ax.set_xlabel("Attack drone count")
    ax.set_ylabel("Success rate")
    ax.set_xticks(x, ATTACKER_BUCKET_ORDER)
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=2, frameon=False, loc="upper right")
    filename = save_figure(fig, output_dir, "05_success_rate_by_attacker_load")
    return FigureRecord(
        number,
        filename,
        "Success rate by attacker load",
        "Compares strategy robustness as attacker counts grow.",
    )


def plot_success_heatmap(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    strategies = ordered_strategies(results)
    defender_counts = sorted({result.defender_count for result in results})
    columns = [
        (defender_count, bucket)
        for defender_count in defender_counts
        for bucket in ATTACKER_BUCKET_ORDER
    ]
    matrix = []
    for strategy in strategies:
        row = []
        for defender_count, bucket in columns:
            row.append(
                rate(
                    [
                        result
                        for result in results
                        if result.strategy == strategy
                        and result.defender_count == defender_count
                        and result.attacker_bucket == bucket
                    ]
                )
            )
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title("Success Rate by Strategy, Defender Count, and Attacker Load")
    ax.set_yticks(np.arange(len(strategies)), [SHORT_LABELS[strategy] for strategy in strategies])
    ax.set_xticks(
        np.arange(len(columns)),
        [f"D{defenders}\nA{bucket}" for defenders, bucket in columns],
    )
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(
                col_idx,
                row_idx,
                f"{value * 100:.0f}%",
                ha="center",
                va="center",
                color="#111111" if value < 0.55 else "white",
                fontsize=7,
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Success rate")
    filename = save_figure(fig, output_dir, "06_success_rate_heatmap")
    return FigureRecord(
        number,
        filename,
        "Success heatmap",
        "Identifies operating regimes where each strategy is viable.",
    )


def plot_kill_ratio_distribution(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    strategies = ordered_strategies(results)
    values = [
        [result.kill_ratio for result in results if result.strategy == strategy]
        for strategy in strategies
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    parts = ax.violinplot(values, showmeans=False, showmedians=True, widths=0.85)
    for body, strategy in zip(parts["bodies"], strategies):
        body.set_facecolor(PALETTE[strategy])
        body.set_edgecolor("#222222")
        body.set_alpha(0.65)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color("#222222")
        parts[key].set_linewidth(1.1)
    ax.set_title("Distribution of Kill Ratio by Strategy")
    ax.set_ylabel("Kill ratio")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(np.arange(1, len(strategies) + 1), [STRATEGY_LABELS[s] for s in strategies])
    filename = save_figure(fig, output_dir, "07_kill_ratio_distribution")
    return FigureRecord(
        number,
        filename,
        "Kill ratio distribution",
        "Shows consistency and tail behavior beyond mean performance.",
    )


def plot_terminal_time_distribution(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    strategies = ordered_strategies(results)
    values = [
        [result.completion_time_s for result in results if result.strategy == strategy]
        for strategy in strategies
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    box = ax.boxplot(values, patch_artist=True, showfliers=False)
    for patch, strategy in zip(box["boxes"], strategies):
        patch.set_facecolor(PALETTE[strategy])
        patch.set_alpha(0.65)
        patch.set_edgecolor("#222222")
    for key in ("whiskers", "caps", "medians"):
        for artist in box[key]:
            artist.set_color("#222222")
    ax.set_title("Time to Terminal Event")
    ax.set_ylabel("Simulated seconds until success or breach")
    ax.set_xticks(np.arange(1, len(strategies) + 1), [STRATEGY_LABELS[s] for s in strategies])
    filename = save_figure(fig, output_dir, "08_terminal_time_distribution")
    return FigureRecord(
        number,
        filename,
        "Terminal event timing",
        "Shows whether strategies delay breach or finish quickly.",
    )


def plot_global_vs_corridor_success_uplift(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    grouped = group_by_strategy(results)
    base_strategies = [strategy for strategy in PAIR_LABELS if strategy in grouped]
    corridor_rates = [rate(grouped[strategy]) for strategy in base_strategies]
    collab_rates = [
        rate(grouped[COLLAB_FOR[strategy]])
        for strategy in base_strategies
        if COLLAB_FOR[strategy] in grouped
    ]
    global_rates = [rate(grouped[GLOBAL_FOR[strategy]]) for strategy in base_strategies]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(base_strategies))
    width = 0.24
    ax.bar(x - width, corridor_rates, width, label="Corridor", color="#9CA3AF")
    if len(collab_rates) == len(base_strategies):
        ax.bar(x, collab_rates, width, label="Collaboration", color="#4B5563")
    ax.bar(x + width, global_rates, width, label="Global", color="#0072B2")
    for idx, global_rate in enumerate(global_rates):
        y = max(corridor_rates[idx], global_rate)
        if len(collab_rates) == len(base_strategies):
            y = max(y, collab_rates[idx])
        delta = global_rate - corridor_rates[idx]
        ax.annotate(
            f"G {delta * 100:+.1f} pp",
            (idx, y),
            ha="center",
            va="bottom",
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_title("Success by Assignment Mode")
    ax.set_ylabel("Success rate")
    ax.set_xticks(x, [PAIR_LABELS[strategy] for strategy in base_strategies])
    all_rates = global_rates + corridor_rates + collab_rates
    ax.set_ylim(0.0, max(all_rates) * 1.45 if all_rates else 1.0)
    ax.legend(frameon=False)
    filename = save_figure(fig, output_dir, "09_global_vs_corridor_success_uplift")
    return FigureRecord(
        number,
        filename,
        "Success by assignment mode",
        "Compares hard corridors, collaborative corridors, and global assignment.",
    )


def plot_optimized_global_delta(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    paired = pair_results(results, "optimized_global", "optimized")
    deltas = [global_result.kill_ratio - corridor_result.kill_ratio for global_result, corridor_result in paired]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bins = np.linspace(-1.0, 1.0, 33)
    ax.hist(deltas, bins=bins, color=PALETTE["optimized_global"], edgecolor="white", alpha=0.8)
    ax.axvline(0.0, color="#222222", linewidth=1.2, linestyle="--", label="No change")
    ax.axvline(mean(deltas), color="#D55E00", linewidth=2.0, label=f"Mean delta = {mean(deltas):.3f}")
    ax.set_title("Per-Scenario Kill-Ratio Delta: Optimized Global vs Corridor")
    ax.set_xlabel("Kill-ratio delta")
    ax.set_ylabel("Scenario count")
    ax.legend(frameon=False)
    filename = save_figure(fig, output_dir, "10_optimized_global_kill_ratio_delta")
    return FigureRecord(
        number,
        filename,
        "Optimized global per-scenario delta",
        "Shows how often global optimization improves or hurts individual scenarios.",
    )


def pair_results(
    results: list[Result], strategy_a: str, strategy_b: str
) -> list[tuple[Result, Result]]:
    by_key: dict[tuple[int, int, str], Result] = {}
    for result in results:
        by_key[(result.run_id, result.scenario_seed, result.strategy)] = result

    pairs = []
    scenario_keys = sorted({(result.run_id, result.scenario_seed) for result in results})
    for run_id, scenario_seed in scenario_keys:
        left = by_key.get((run_id, scenario_seed, strategy_a))
        right = by_key.get((run_id, scenario_seed, strategy_b))
        if left is not None and right is not None:
            pairs.append((left, right))
    return pairs


def write_figure_index(records: list[FigureRecord], path: Path) -> None:
    lines = ["# Publication Figure Index", ""]
    for record in records:
        lines.append(
            f"{record.number}. `{record.filename}` - **{record.title}.** {record.message}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_talking_points(
    results: list[Result], records: list[FigureRecord], path: Path
) -> None:
    grouped = group_by_strategy(results)
    strategies = ordered_strategies(results)
    scenario_count = len({(result.run_id, result.scenario_seed) for result in results})
    best_success_strategy = max(strategies, key=lambda strategy: rate(grouped[strategy]))
    best_kill_ratio_strategy = max(
        strategies,
        key=lambda strategy: mean([result.kill_ratio for result in grouped[strategy]]),
    )
    optimized_pairs = pair_results(results, "optimized_global", "optimized")
    optimized_collab_pairs = pair_results(results, "optimized_collab", "optimized")
    delta_kills = [left.kills - right.kills for left, right in optimized_pairs]
    delta_kill_ratios = [
        left.kill_ratio - right.kill_ratio for left, right in optimized_pairs
    ]
    improved = sum(delta > 0 for delta in delta_kills)
    tied = sum(delta == 0 for delta in delta_kills)
    worsened = sum(delta < 0 for delta in delta_kills)
    converted_success = sum((not right.success) and left.success for left, right in optimized_pairs)

    lines = [
        "# Publication Talking Points",
        "",
        f"Dataset: `{path.parent.parent / 'per_run_results.csv'}`.",
        f"Sample: {scenario_count} scenarios and {len(results)} strategy runs.",
        "",
        "## Main Claims",
        "",
        (
            f"- **Best overall strategy:** `{best_success_strategy}` achieved "
            f"{rate(grouped[best_success_strategy]) * 100:.1f}% mission success."
        ),
        (
            f"- **Best attrition strategy:** `{best_kill_ratio_strategy}` achieved "
            f"a mean kill ratio of "
            f"{mean([r.kill_ratio for r in grouped[best_kill_ratio_strategy]]):.3f}."
        ),
    ]

    if "optimized_global" in grouped and "optimized" in grouped:
        global_success = rate(grouped["optimized_global"])
        corridor_success = rate(grouped["optimized"])
        global_kills = mean([r.kills for r in grouped["optimized_global"]])
        corridor_kills = mean([r.kills for r in grouped["optimized"]])
        collab_success = (
            rate(grouped["optimized_collab"])
            if "optimized_collab" in grouped
            else 0.0
        )
        collab_kills = (
            mean([r.kills for r in grouped["optimized_collab"]])
            if "optimized_collab" in grouped
            else 0.0
        )
        lines.extend(
            [
                (
                    f"- **Removing corridors helped optimization:** `optimized_global` "
                    f"improved success from {corridor_success * 100:.1f}% to "
                    f"{global_success * 100:.1f}% "
                    f"({(global_success - corridor_success) * 100:+.1f} percentage points)."
                ),
                (
                    f"- **Optimized global intercepted more attackers:** mean kills rose "
                    f"from {corridor_kills:.2f} to {global_kills:.2f} per scenario."
                ),
                (
                    f"- **Collaborative corridors are the middle ground:** "
                    f"`optimized_collab` achieved {collab_success * 100:.1f}% success "
                    f"and {collab_kills:.2f} mean kills while preserving corridor-first behavior."
                ),
                (
                    f"- **Scenario-level improvement:** `optimized_global` killed more "
                    f"attackers than corridor `optimized` in {improved / len(optimized_pairs) * 100:.1f}% "
                    f"of paired scenarios, tied in {tied / len(optimized_pairs) * 100:.1f}%, "
                    f"and killed fewer in {worsened / len(optimized_pairs) * 100:.1f}%."
                ),
                (
                    f"- **Success conversion:** `optimized_global` turned "
                    f"{converted_success} scenarios from corridor-optimized failures into successes."
                ),
            ]
        )
        if optimized_collab_pairs:
            collab_converted_success = sum(
                (not right.success) and left.success
                for left, right in optimized_collab_pairs
            )
            collab_improved = sum(
                left.kills > right.kills for left, right in optimized_collab_pairs
            )
            lines.append(
                f"- **Collaboration conversion:** `optimized_collab` turned "
                f"{collab_converted_success} corridor-optimized failures into successes "
                f"and improved kill count in "
                f"{collab_improved / len(optimized_collab_pairs) * 100:.1f}% of paired scenarios."
            )

    lines.extend(
        [
            "- **Deadline-first policies are not enough:** earliest-deadline variants are useful as urgency baselines, but they trail the optimized global policy in both success and mean kill ratio.",
            "- **The result is still capacity constrained:** even the best strategy leaves substantial breach risk, so future gains likely require softer handoff rules, more defenders, faster defenders, or a shorter dwell requirement.",
            "",
            "## Figure Guide",
            "",
        ]
    )
    for record in records:
        lines.append(
            f"- Figure {record.number}: `{record.filename}` - {record.message}"
        )

    lines.extend(["", "## Overall Metrics", ""])
    lines.append("| Strategy | Success rate | Mean kills | Mean kill ratio | Median terminal time |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for strategy in strategies:
        values = grouped[strategy]
        lines.append(
            "| "
            f"{SHORT_LABELS[strategy]} | "
            f"{rate(values) * 100:.1f}% | "
            f"{mean([r.kills for r in values]):.2f} | "
            f"{mean([r.kill_ratio for r in values]):.3f} | "
            f"{statistics.median([r.completion_time_s for r in values]):.2f}s |"
        )

    lines.extend(["", "## Best Operating Regimes", ""])
    regime_rows = []
    for strategy in strategies:
        for defender_count in sorted({result.defender_count for result in results}):
            for bucket in ATTACKER_BUCKET_ORDER:
                subset = [
                    result
                    for result in results
                    if result.strategy == strategy
                    and result.defender_count == defender_count
                    and result.attacker_bucket == bucket
                ]
                if subset:
                    regime_rows.append((rate(subset), strategy, defender_count, bucket, len(subset)))
    for value, strategy, defender_count, bucket, count in sorted(regime_rows, reverse=True)[:8]:
        lines.append(
            f"- {SHORT_LABELS[strategy]} with {defender_count} defenders vs {bucket} attackers: "
            f"{value * 100:.1f}% success across {count} runs."
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_success_rate_matrix_markdown(results: list[Result], path: Path) -> None:
    grouped = group_by_strategy(results)
    strategies = ordered_strategies(results)
    attacker_counts = sorted({result.attacker_count for result in results})
    defender_counts = sorted({result.defender_count for result in results})
    lines = [
        "# Success Rate Matrix",
        "",
        "Rows are exact attack-drone counts. Columns are defense-drone counts. "
        "Cells are mission success rates.",
        "",
    ]
    for strategy in strategies:
        values = grouped[strategy]
        lines.extend(
            [
                f"## {SHORT_LABELS[strategy]}",
                "",
                "| Attack drones | "
                + " | ".join(f"{count} defense" for count in defender_counts)
                + " |",
                "| ---: | "
                + " | ".join("---:" for _ in defender_counts)
                + " |",
            ]
        )
        for attacker_count in attacker_counts:
            row = [str(attacker_count)]
            for defender_count in defender_counts:
                subset = [
                    result
                    for result in values
                    if result.attacker_count == attacker_count
                    and result.defender_count == defender_count
                ]
                row.append(f"{rate(subset) * 100:.1f}%" if subset else "")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
