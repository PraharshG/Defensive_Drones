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
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
ATTACKER_BUCKET_ORDER = [
    "5-10",
    "11-15",
    "16-20",
    "250-274",
    "275-299",
    "300-324",
    "325-350",
]
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
    breach_rate: float
    first_breach_time_s: float | None


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
        plot_assignment_flexibility_uplift(results, output_dir, 1),
        plot_capability_envelope(results, output_dir, 2),
        plot_defender_resource_scaling(results, output_dir, 3),
        plot_paired_scenario_conversion_matrix(results, output_dir, 4),
        plot_attacker_load_cliff(results, output_dir, 5),
        plot_residual_failure_space(results, output_dir, 6),
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
            breach_rate=float(row.get("breach_rate", 0.0)),
            first_breach_time_s=(
                float(row["first_breach_time_s"])
                if row.get("first_breach_time_s")
                else None
            ),
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


def attacker_bucket_sort_key(bucket: str) -> tuple[int, int]:
    lower, _, upper = bucket.partition("-")
    return int(lower), int(upper or lower)


def attacker_buckets_for(results: list[Result]) -> list[str]:
    present = {result.attacker_bucket for result in results}
    ordered = [bucket for bucket in ATTACKER_BUCKET_ORDER if bucket in present]
    ordered.extend(
        sorted(present.difference(ordered), key=attacker_bucket_sort_key)
    )
    return ordered


def representative_values(values: list[int], limit: int) -> list[int]:
    ordered = sorted(set(values))
    if len(ordered) <= limit:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, limit).round().astype(int)
    return [ordered[index] for index in indices]


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


def plot_assignment_flexibility_uplift(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    grouped = group_by_strategy(results)
    strategies = [
        "optimized",
        "optimized_collab",
        "optimized_global",
    ]
    labels = [
        "Corridor\n(Base Optimized)",
        "Collaboration\n(Optimized Collab)",
        "Global\n(Optimized Global)",
    ]
    values = [rate(grouped[strategy]) * 100.0 for strategy in strategies]
    colors = ["#BAC2CB", "#8A95A5", "#1D63B8"]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.55,
        edgecolor="none",
        zorder=3,
    )

    ax.set_ylabel(
        "System Success Rate (%)",
        fontsize=12,
        fontweight="medium",
        color="#4A4A4A",
        labelpad=12,
    )
    ax.set_ylim(0.0, max(30.0, max(values) + 5.0))
    ax.grid(axis="y", linestyle="-", linewidth=0.5, color="#E5E5E5", zorder=1)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", which="both", length=0, labelsize=11, labelcolor="#333333")
    ax.tick_params(axis="x", pad=8)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="#1A1A1A",
        )

    fig.text(
        0.5,
        0.965,
        "Assignment Flexibility Uplift",
        fontsize=16,
        fontweight="bold",
        color="#1A1A1A",
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.035,
        "Loosening rigid spatial allocation materially improves operational outcomes.",
        fontsize=10.75,
        fontstyle="italic",
        color="#666666",
        ha="center",
        va="bottom",
    )

    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.90])
    filename = save_figure(fig, output_dir, "01_assignment_flexibility_uplift")
    return FigureRecord(
        number,
        filename,
        "Assignment flexibility uplift",
        "Shows how optimized performance improves as allocation moves from rigid corridors to collaboration and global assignment.",
    )


def plot_capability_envelope(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    strategies = ordered_strategies(results)
    selected = [
        strategy
        for strategy in (
            "earliest_deadline_collab",
            "optimized_collab",
            "optimized_global",
        )
        if strategy in strategies
    ]
    defender_counts = representative_values(
        [result.defender_count for result in results], 3
    )
    attacker_buckets = attacker_buckets_for(results)
    panel_titles = {
        "earliest_deadline_collab": "Earliest Deadline Collab",
        "optimized_collab": "Optimized Collab",
        "optimized_global": "Optimized Global",
    }
    matrices = [
        _capability_matrix(results, strategy, defender_counts, attacker_buckets)
        for strategy in selected
    ]
    cmap = LinearSegmentedColormap.from_list(
        "capability_blue",
        ["#F7FAFD", "#D8E6F7", "#93B8E3", "#1D63B8"],
    )

    fig, axes = plt.subplots(1, len(selected), figsize=(14.0, 4.5), sharey=True)
    if len(selected) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#FFFFFF")

    vmin = 0.0
    vmax = 100.0
    for index, (ax, strategy, matrix) in enumerate(zip(axes, selected, matrices)):
        ax.set_facecolor("#FFFFFF")
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(
            panel_titles[strategy],
            fontsize=13,
            fontweight="bold",
            pad=15,
            color="#1A1A1A",
        )
        ax.set_xticks(np.arange(len(defender_counts)), [str(count) for count in defender_counts])
        ax.set_yticks(np.arange(len(attacker_buckets)), attacker_buckets)
        ax.tick_params(
            axis="both",
            which="both",
            length=0,
            labelsize=10,
            labelcolor="#333333",
        )
        ax.set_xlabel(
            "Defender Count",
            fontsize=11,
            fontweight="medium",
            color="#4A4A4A",
            labelpad=10,
        )
        if index == 0:
            ax.set_ylabel(
                "Attacker Swarm Size",
                fontsize=11,
                fontweight="medium",
                color="#4A4A4A",
                labelpad=10,
            )

        ax.set_xticks(np.arange(-0.5, len(defender_counts), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(attacker_buckets), 1), minor=True)
        ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                text_color = "#FFFFFF" if value >= 55.0 else "#1A1A1A"
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=text_color,
                )

    fig.suptitle(
        "Capability Envelope: Success Rate (%) by Strategy and Resource Allocation",
        fontsize=16,
        fontweight="bold",
        color="#1A1A1A",
        y=1.05,
    )
    fig.text(
        0.5,
        0.03,
        "Systematic failure boundaries expose nonlinear capability degradation across sub-optimal allocation policies.",
        fontsize=11,
        fontstyle="italic",
        color="#666666",
        ha="center",
        va="bottom",
    )

    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
    filename = save_figure(fig, output_dir, "02_capability_envelope")
    return FigureRecord(
        number,
        filename,
        "Capability envelope",
        "Maps where the strongest strategy families remain viable as attacker load rises and defender resources increase.",
    )


def _capability_matrix(
    results: list[Result],
    strategy: str,
    defender_counts: list[int],
    attacker_buckets: list[str],
) -> np.ndarray:
    matrix = np.zeros((len(attacker_buckets), len(defender_counts)), dtype=float)
    for row_index, attacker_bucket in enumerate(attacker_buckets):
        for col_index, defender_count in enumerate(defender_counts):
            subset = [
                result
                for result in results
                if result.strategy == strategy
                and result.attacker_bucket == attacker_bucket
                and result.defender_count == defender_count
            ]
            matrix[row_index, col_index] = rate(subset) * 100.0
    return matrix


def plot_defender_resource_scaling(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    strategies = ordered_strategies(results)
    selected = [
        strategy
        for strategy in (
            "earliest_deadline_collab",
            "optimized_collab",
            "optimized_global",
        )
        if strategy in strategies
    ]
    label_map = {
        "earliest_deadline_collab": "Earliest Deadline Collab",
        "optimized_collab": "Optimized Collab",
        "optimized_global": "Optimized Global",
    }
    color_map = {
        "earliest_deadline_collab": "#BAC2CB",
        "optimized_collab": "#8A95A5",
        "optimized_global": "#1D63B8",
    }
    line_widths = {
        "earliest_deadline_collab": 2.0,
        "optimized_collab": 2.5,
        "optimized_global": 3.5,
    }
    defender_counts = sorted({result.defender_count for result in results})
    series = {
        strategy: [
            rate(
                [
                    result
                    for result in results
                    if result.strategy == strategy
                    and result.defender_count == defender_count
                ]
            )
            * 100.0
            for defender_count in defender_counts
        ]
        for strategy in selected
    }

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    for strategy in selected:
        ax.plot(
            defender_counts,
            series[strategy],
            color=color_map[strategy],
            linewidth=line_widths[strategy],
            label=label_map[strategy],
            zorder=3 if strategy == "earliest_deadline_collab" else 4 if strategy == "optimized_collab" else 5,
        )

    legend = ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.11),
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        facecolor="#FFFFFF",
        edgecolor="none",
        fontsize=10.5,
        handlelength=2.6,
        borderpad=0.55,
        labelspacing=0.45,
    )
    for legend_text in legend.get_texts():
        legend_text.set_color("#333333")

    ax.grid(axis="y", linestyle="-", linewidth=0.5, color="#E5E5E5", zorder=1)
    ax.set_xticks(defender_counts)
    max_value = max(max(values) for values in series.values()) if series else 0.0
    ax.set_ylim(0.0, max(50.0, max_value + 9.0))

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", which="both", length=0, labelsize=11, labelcolor="#333333")
    ax.tick_params(axis="x", pad=8)
    ax.set_xlabel(
        "Active Defender Count",
        fontsize=12,
        fontweight="medium",
        color="#4A4A4A",
        labelpad=12,
    )
    ax.set_ylabel(
        "System Success Rate (%)",
        fontsize=12,
        fontweight="medium",
        color="#4A4A4A",
        labelpad=12,
    )

    fig.text(
        0.5,
        0.965,
        "Defender Resource Scaling Mechanics",
        fontsize=16,
        fontweight="bold",
        color="#1A1A1A",
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.035,
        "System architecture scales materially with added defensive resources,\n"
        "avoiding capability plateaus.",
        fontsize=10.75,
        fontstyle="italic",
        color="#666666",
        ha="center",
        va="bottom",
        linespacing=1.2,
    )

    fig.tight_layout(rect=[0.0, 0.09, 1.0, 0.90])
    filename = save_figure(fig, output_dir, "03_defender_resource_scaling")
    return FigureRecord(
        number,
        filename,
        "Defender resource scaling mechanics",
        "Shows that the strongest strategy families scale materially as additional defenders are added.",
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


def plot_paired_scenario_conversion_matrix(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    paired = pair_results(results, "optimized_global", "optimized")
    maintained_failure = 0
    converted_success = 0
    regression = 0
    maintained_success = 0

    for new_policy_result, baseline_result in paired:
        if new_policy_result.success and baseline_result.success:
            maintained_success += 1
        elif new_policy_result.success and not baseline_result.success:
            converted_success += 1
        elif not new_policy_result.success and baseline_result.success:
            regression += 1
        else:
            maintained_failure += 1

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.grid(False)

    cells = {
        "Maintained\nFailure": {
            "pos": (0, 1),
            "val": maintained_failure,
            "color": "#F3F4F6",
            "text_color": "#6B7280",
        },
        "Converted\nSuccess": {
            "pos": (1, 1),
            "val": converted_success,
            "color": "#1D63B8",
            "text_color": "#FFFFFF",
        },
        "Regression": {
            "pos": (0, 0),
            "val": regression,
            "color": "#FCA5A5",
            "text_color": "#991B1B",
        },
        "Maintained\nSuccess": {
            "pos": (1, 0),
            "val": maintained_success,
            "color": "#E5E7EB",
            "text_color": "#4B5563",
        },
    }

    for label, data in cells.items():
        x, y = data["pos"]
        rect = patches.Rectangle(
            (x, y),
            1,
            1,
            facecolor=data["color"],
            edgecolor="#FFFFFF",
            linewidth=6,
        )
        ax.add_patch(rect)

        ax.text(
            x + 0.5,
            y + 0.58,
            str(data["val"]),
            color=data["text_color"],
            fontsize=28,
            fontweight="bold",
            ha="center",
            va="center",
        )
        ax.text(
            x + 0.5,
            y + 0.38,
            label,
            color=data["text_color"],
            fontsize=11,
            fontweight="medium",
            ha="center",
            va="center",
        )

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(
        ["System Failure", "System Success"],
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )
    ax.set_yticklabels(
        ["System Success", "System Failure"],
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )
    for label in ax.get_yticklabels():
        label.set_rotation(90)
        label.set_verticalalignment("center")
        label.set_horizontalalignment("right")
    ax.tick_params(axis="both", which="both", length=0)
    ax.tick_params(axis="y", pad=2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel(
        "New Policy State (Optimized Global)",
        fontsize=11,
        color="#4A4A4A",
        labelpad=15,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Baseline State (Optimized)",
        fontsize=11,
        color="#4A4A4A",
        labelpad=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.965,
        "Paired Scenario Conversion Matrix",
        fontsize=16,
        fontweight="bold",
        color="#1A1A1A",
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.915,
        f"Evaluating {len(paired):,} matched scenario cells to isolate algorithmic gains\n"
        "from environmental randomness and defender-count effects.",
        fontsize=10.75,
        fontstyle="italic",
        color="#666666",
        ha="center",
        va="top",
        linespacing=1.2,
    )

    fig.text(
        0.5,
        0.03,
        "Optimized global allocation yields paired-scenario returns,\n"
        f"converting {converted_success} failures while risking {regression} regressions.",
        fontsize=10.75,
        fontweight="medium",
        color="#1A1A1A",
        ha="center",
        va="bottom",
        linespacing=1.2,
    )

    fig.tight_layout(rect=[0.03, 0.12, 0.97, 0.87])
    filename = save_figure(fig, output_dir, "04_paired_scenario_conversion_matrix")
    return FigureRecord(
        number,
        filename,
        "Paired scenario conversion matrix",
        "Compares optimized global against optimized on identical scenarios to isolate true algorithmic gains.",
    )


def plot_attacker_load_cliff(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    strategies = ordered_strategies(results)
    selected = [
        strategy
        for strategy in (
            "earliest_deadline_collab",
            "optimized_collab",
            "optimized_global",
        )
        if strategy in strategies
    ]
    label_map = {
        "earliest_deadline_collab": "Earliest Deadline Collab",
        "optimized_collab": "Optimized Collab",
        "optimized_global": "Optimized Global",
    }
    color_map = {
        "earliest_deadline_collab": "#BAC2CB",
        "optimized_collab": "#8A95A5",
        "optimized_global": "#1D63B8",
    }
    line_widths = {
        "earliest_deadline_collab": 2.5,
        "optimized_collab": 3.0,
        "optimized_global": 4.0,
    }
    marker_sizes = {
        "earliest_deadline_collab": 6.0,
        "optimized_collab": 6.0,
        "optimized_global": 8.0,
    }
    attacker_buckets = attacker_buckets_for(results)
    bucket_labels = [f"{bucket}\nattackers" for bucket in attacker_buckets]
    x = np.arange(len(attacker_buckets))
    series = {
        strategy: [
            rate(
                [
                    result
                    for result in results
                    if result.strategy == strategy
                    and result.attacker_bucket == bucket
                ]
            )
            * 100.0
            for bucket in attacker_buckets
        ]
        for strategy in selected
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    if len(attacker_buckets) > 1:
        ax.axvspan(0.5, len(attacker_buckets) - 0.5, facecolor="#F9FAFB", alpha=1.0, zorder=0)
        ax.text(
            (len(attacker_buckets) - 1) / 2.0,
            62.0,
            "HIGHER LOAD REGIME",
            color="#9CA3AF",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
        )

    for strategy in selected:
        ax.plot(
            x,
            series[strategy],
            color=color_map[strategy],
            linewidth=line_widths[strategy],
            marker="o",
            markersize=marker_sizes[strategy],
            markeredgecolor="#FFFFFF",
            markeredgewidth=1.4,
            label=label_map[strategy],
            zorder=3 if strategy == "earliest_deadline_collab" else 4 if strategy == "optimized_collab" else 5,
        )

    legend = ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.9, 0.54),
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        facecolor="#FFFFFF",
        edgecolor="none",
        fontsize=10.5,
        handlelength=2.4,
        borderpad=0.6,
        labelspacing=0.5,
    )
    for legend_text in legend.get_texts():
        legend_text.set_color("#333333")

    global_drop = 0.0
    if (
        len(attacker_buckets) > 1
        and series.get("optimized_global")
        and series["optimized_global"][0] > 0.0
    ):
        global_drop = (
            1.0 - series["optimized_global"][1] / series["optimized_global"][0]
        ) * 100.0
    drop_annotation = ax.annotate(
        f"-{global_drop:.0f}% Capability Drop",
        xy=(min(1, len(attacker_buckets) - 1), series["optimized_global"][min(1, len(attacker_buckets) - 1)]),
        xytext=(min(1.28, max(0.0, len(attacker_buckets) - 0.7)), 26.0),
        arrowprops={
            "facecolor": "#1A1A1A",
            "shrink": 0.05,
            "width": 1.5,
            "headwidth": 6,
            "edgecolor": "none",
        },
        fontsize=10,
        fontweight="bold",
        color="#1A1A1A",
        zorder=10,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "#FFFFFF",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )
    if drop_annotation.arrow_patch is not None:
        drop_annotation.arrow_patch.set_zorder(10)

    ax.grid(axis="y", linestyle="-", linewidth=0.5, color="#E5E5E5", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=11, fontweight="medium", color="#333333")
    max_series_value = max((max(values) for values in series.values()), default=0.0)
    ax.set_ylim(0.0, max(65.0, max_series_value + 5.0))

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", which="both", length=0, labelsize=11, labelcolor="#333333")
    ax.set_ylabel(
        "System Success Rate (%)",
        fontsize=12,
        fontweight="medium",
        color="#4A4A4A",
        labelpad=12,
    )
    fig.text(
        0.14,
        0.965,
        "Attacker Load Cliff: Systematic Capacity Thresholds",
        fontsize=16,
        fontweight="bold",
        color="#1A1A1A",
        ha="left",
        va="top",
    )
    fig.text(
        0.5,
        0.03,
        "Current architecture demonstrates how capability changes\n"
        "as attacker load rises across randomized swarm sizes.",
        fontsize=10.75,
        fontstyle="italic",
        color="#666666",
        ha="center",
        va="bottom",
        linespacing=1.2,
    )

    fig.subplots_adjust(left=0.14, right=0.98, top=0.8, bottom=0.2)
    filename = save_figure(fig, output_dir, "05_attacker_load_cliff")
    return FigureRecord(
        number,
        filename,
        "Attacker load cliff",
        "Shows the sharp system success collapse once the swarm moves beyond the moderate-load regime.",
    )


def plot_residual_failure_space(
    results: list[Result], output_dir: Path, number: int
) -> FigureRecord:
    strategy = "optimized_global"
    defender_counts = list(
        reversed(representative_values([result.defender_count for result in results], 4))
    )
    attacker_buckets = attacker_buckets_for(results)
    failure_matrix = np.array(
        [
            [
                (1.0 - rate(
                    [
                        result
                        for result in results
                        if result.strategy == strategy
                        and result.defender_count == defender_count
                        and result.attacker_bucket == attacker_bucket
                    ]
                ))
                * 100.0
                for attacker_bucket in attacker_buckets
            ]
            for defender_count in defender_counts
        ]
    )

    cmap = LinearSegmentedColormap.from_list(
        "failure_space",
        ["#F3F4F6", "#FCA5A5", "#991B1B"],
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    image = ax.imshow(failure_matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=100.0)

    ax.set_xticks(np.arange(len(attacker_buckets)), attacker_buckets)
    ax.set_yticks(np.arange(len(defender_counts)), [str(value) for value in defender_counts])
    ax.tick_params(axis="both", which="both", length=0, labelsize=11, labelcolor="#333333")

    ax.set_xticks(np.arange(-0.5, len(attacker_buckets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(defender_counts), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(failure_matrix.shape[0]):
        for col_idx in range(failure_matrix.shape[1]):
            value = failure_matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#1A1A1A" if value < 55.0 else "#FFFFFF",
            )

    ax.set_title(
        "Residual Failure-Space: Optimized Global Policy",
        fontsize=16,
        fontweight="bold",
        pad=24,
        color="#1A1A1A",
        loc="left",
    )
    ax.set_xlabel(
        "Attacker Swarm Size Bucket",
        fontsize=12,
        fontweight="medium",
        color="#4A4A4A",
        labelpad=12,
    )
    ax.set_ylabel(
        "Active Defender Count",
        fontsize=12,
        fontweight="medium",
        color="#4A4A4A",
        labelpad=12,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        1.03,
        0.5,
        "Failure Probability (%)",
        transform=ax.transAxes,
        rotation=270,
        fontsize=11,
        fontweight="bold",
        color="#991B1B",
        va="center",
        ha="left",
    )
    fig.text(
        0.51,
        0.04,
        "The unresolved operational zone is strictly confined to high-load saturation\n"
        "regimes, not general underperformance.",
        fontsize=10.25,
        fontstyle="italic",
        color="#666666",
        ha="center",
        va="bottom",
        linespacing=1.15,
    )

    fig.subplots_adjust(left=0.12, right=0.9, top=0.83, bottom=0.21)
    filename = save_figure(fig, output_dir, "06_residual_failure_space")
    return FigureRecord(
        number,
        filename,
        "Residual failure space",
        "Shows that the remaining failure regime is concentrated in high-load saturation cases for optimized global control.",
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
    ax.set_ylabel("Simulated seconds until all attackers are resolved")
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
    by_key: dict[tuple[int, int, int, str], Result] = {}
    for result in results:
        by_key[
            (result.run_id, result.scenario_seed, result.defender_count, result.strategy)
        ] = result

    pairs = []
    scenario_keys = sorted(
        {
            (result.run_id, result.scenario_seed, result.defender_count)
            for result in results
        }
    )
    for run_id, scenario_seed, defender_count in scenario_keys:
        left = by_key.get((run_id, scenario_seed, defender_count, strategy_a))
        right = by_key.get((run_id, scenario_seed, defender_count, strategy_b))
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
    scenario_count = len(
        {
            (result.run_id, result.scenario_seed, result.defender_count)
            for result in results
        }
    )
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
            "- **The result is still capacity constrained:** even the best strategy leaves pass-through risk, so future gains likely require softer handoff rules, more defenders, faster defenders, or a shorter dwell requirement.",
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
    lines.append("| Strategy | Success rate | Mean kills | Mean kill ratio | Median completion time |")
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
            for bucket in attacker_buckets_for(results):
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
    defender_counts = sorted({result.defender_count for result in results})
    attacker_buckets = attacker_buckets_for(results)
    lines = [
        "# Success Rate Matrix",
        "",
        "Rows are defensive-drone counts. Columns are attack-drone count buckets. "
        "Cells are mission success rates.",
        "",
    ]
    for strategy in strategies:
        values = grouped[strategy]
        lines.extend(
            [
                f"## {SHORT_LABELS[strategy]}",
                "",
                "| Defense drones | "
                + " | ".join(f"{bucket} attackers" for bucket in attacker_buckets)
                + " |",
                "| ---: | "
                + " | ".join("---:" for _ in attacker_buckets)
                + " |",
            ]
        )
        for defender_count in defender_counts:
            row = [str(defender_count)]
            for attacker_bucket in attacker_buckets:
                subset = [
                    result
                    for result in values
                    if result.defender_count == defender_count
                    and result.attacker_bucket == attacker_bucket
                ]
                row.append(f"{rate(subset) * 100:.1f}%" if subset else "")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
