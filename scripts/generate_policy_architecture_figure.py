from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from generate_publication_figures import configure_matplotlib, save_figure


ROWS = [
    (
        "optimized",
        "Optimized",
        "Weighted matching",
        "Ranks by intercept time\nand deadline risk.",
        "#0F6CBD",
    ),
    (
        "nearest",
        "Nearest",
        "Distance greedy",
        "Chooses the closest\nlive attacker first.",
        "#1F9D73",
    ),
    (
        "earliest_deadline",
        "Earliest Deadline",
        "Urgency greedy",
        "Chooses the shortest\ntime-to-target attacker.",
        "#D97706",
    ),
]

REGIMES = [
    (
        "corridor",
        "Corridor",
        "Sector ownership",
        "Defenders stay inside their\nassigned angular sector.",
        "#9CA3AF",
        "#F3F4F6",
    ),
    (
        "collaboration",
        "Collaboration",
        "Shared assists",
        "Corridor-first, then idle\nneighbors help overloads.",
        "#475569",
        "#EEF2F7",
    ),
    (
        "global",
        "Global",
        "Shared global pool",
        "Any defender can engage any\nlive attacker.",
        "#1D63B8",
        "#EAF2FF",
    ),
    (
        "autonomy",
        "Autonomy",
        "Independent local control",
        "No shared state between\ndefenders after launch.",
        "#9A3412",
        "#FFF1E8",
    ),
]

CELL_TEXT = {
    ("optimized", "corridor"): (
        "Optimized Corridor",
        "Scores threats inside the owned\nsector with a near-capture lock.",
    ),
    ("optimized", "collaboration"): (
        "Optimized Collab",
        "Runs corridor matching first,\nthen lets idle neighbors\nassist overloads.",
    ),
    ("optimized", "global"): (
        "Optimized Global",
        "Uses the same scoring with a\nfully shared matching pool.",
    ),
    ("optimized", "autonomy"): (
        "Optimized Autonomy",
        "Each defender applies the score\nlocally from its own\nobservations.",
    ),
    ("nearest", "corridor"): (
        "Nearest Corridor",
        "Greedy nearest-target pursuit\nwithin the owned sector.",
    ),
    ("nearest", "collaboration"): (
        "Nearest Collab",
        "Corridor-first nearest rule,\nthen idle neighbors help\nnearby overloads.",
    ),
    ("nearest", "global"): (
        "Nearest Global",
        "Greedy nearest-target pursuit\nacross the full\nattacker set.",
    ),
    ("nearest", "autonomy"): (
        "Nearest Autonomy",
        "Each defender independently\nchases its nearest\nperceived attacker.",
    ),
    ("earliest_deadline", "corridor"): (
        "Deadline Corridor",
        "Targets the most urgent attacker\ninside the owned sector.",
    ),
    ("earliest_deadline", "collaboration"): (
        "Deadline Collab",
        "Applies urgency inside corridor,\nthen assists the most\nurgent leftovers.",
    ),
    ("earliest_deadline", "global"): (
        "Deadline Global",
        "Urgency-first targeting across\nthe full shared\nattacker set.",
    ),
    ("earliest_deadline", "autonomy"): (
        "Deadline Autonomy",
        "Each defender independently\nselects the most urgent\nperceived threat.",
    ),
}

SUMMARY_CARDS = [
    (
        "Update Highlight",
        "Adds Autonomy as a fourth regime.\nSeparates shared and\nno-shared-state control.",
        "#111827",
        "#F8FAFC",
    ),
    (
        "Measurement",
        "Primary metric: attackers that\nreach the target\n(missed or breached).",
        "#B42318",
        "#FEF3F2",
    ),
    (
        "Scenario Presets",
        "300 attackers; 10-100 defenders\n(step 10);\nstart range about 5 miles.",
        "#1D4ED8",
        "#EFF6FF",
    ),
    (
        "Variable Knobs",
        "Attacker speed, defender speed,\nattack load, and\ndistance from target.",
        "#047857",
        "#ECFDF5",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the updated defensive-drone policy architecture figure."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/publication_figures"),
        help="Directory for PNG/PDF outputs.",
    )
    parser.add_argument(
        "--filename",
        default="12_policy_architecture_matrix_v2",
        help="Base filename for the rendered figure.",
    )
    return parser


def draw_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    edgecolor: str,
    facecolor: str,
    title: str,
    body: str,
    title_size: float,
    body_size: float,
    accent_width: float = 0.0,
) -> None:
    card = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.3,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(card)
    if accent_width > 0.0:
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                accent_width,
                height,
                linewidth=0,
                facecolor=edgecolor,
            )
        )
    text_x = x + accent_width + 0.14
    ax.text(
        text_x,
        y + height - 0.19,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        text_x,
        y + 0.2,
        body,
        ha="left",
        va="bottom",
        fontsize=body_size,
        color="#374151",
    )


def draw_row_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    edgecolor: str,
    title: str,
    subtitle: str,
    body: str,
) -> None:
    card = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.3,
        edgecolor=edgecolor,
        facecolor="#FFFFFF",
    )
    ax.add_patch(card)
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            0.16,
            height,
            linewidth=0,
            facecolor=edgecolor,
        )
    )
    text_x = x + 0.3
    ax.text(
        text_x,
        y + height - 0.2,
        title,
        ha="left",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        text_x,
        y + height - 0.48,
        subtitle,
        ha="left",
        va="top",
        fontsize=8.6,
        color=edgecolor,
    )
    ax.text(
        text_x,
        y + 0.18,
        body,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#374151",
    )


def draw_policy_figure(output_dir: Path, filename: str) -> str:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16.0, 7.9))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0.0, 16.8)
    ax.set_ylim(0.0, 10.8)
    ax.axis("off")

    fig.text(
        0.5,
        0.965,
        "Defensive Drone Policy Matrix",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.932,
        "Updated view: autonomy is separated from shared-state regimes, and the scenario knobs are called out explicitly.",
        ha="center",
        va="top",
        fontsize=10.8,
        color="#475467",
    )

    row_x = 0.7
    row_w = 2.8
    grid_x = 3.8
    cell_w = 2.95
    cell_h = 1.6
    gap_x = 0.2
    gap_y = 0.24
    header_y = 8.05
    header_h = 1.18
    top_y = 6.3

    shared_width = 3 * cell_w + 2 * gap_x
    autonomy_x = grid_x + 3 * (cell_w + gap_x)
    group_y = 9.45
    group_h = 0.48
    shared_band = patches.FancyBboxPatch(
        (grid_x, group_y),
        shared_width,
        group_h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=0,
        facecolor="#E6EEF8",
    )
    autonomy_band = patches.FancyBboxPatch(
        (autonomy_x, group_y),
        cell_w,
        group_h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=0,
        facecolor="#FDE7D8",
    )
    ax.add_patch(shared_band)
    ax.add_patch(autonomy_band)
    ax.text(
        grid_x + shared_width / 2.0,
        group_y + group_h / 2.0,
        "Info shared across defenders",
        ha="center",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color="#1F2937",
    )
    ax.text(
        autonomy_x + cell_w / 2.0,
        group_y + group_h / 2.0,
        "No info shared",
        ha="center",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color="#7C2D12",
    )

    ax.text(
        grid_x + (4 * cell_w + 3 * gap_x) / 2.0,
        10.18,
        "Assignment and Control Regimes",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        row_x + row_w / 2.0,
        10.18,
        "Targeting Policies",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#111827",
    )

    for col_index, (col_key, title, _subtitle, blurb, header_color, body_color) in enumerate(REGIMES):
        x = grid_x + col_index * (cell_w + gap_x)
        header = patches.FancyBboxPatch(
            (x, header_y),
            cell_w,
            header_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0,
            facecolor=header_color,
        )
        ax.add_patch(header)
        ax.text(
            x + 0.14,
            header_y + header_h - 0.18,
            title,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="#FFFFFF",
        )
        ax.text(
            x + 0.14,
            header_y + 0.2,
            blurb,
            ha="left",
            va="bottom",
            fontsize=8.4,
            color="#F8FAFC",
        )
        if col_key == "autonomy":
            badge = patches.FancyBboxPatch(
                (x + cell_w - 0.84, header_y + header_h - 0.38),
                0.68,
                0.24,
                boxstyle="round,pad=0.01,rounding_size=0.06",
                linewidth=0,
                facecolor="#FED7AA",
            )
            ax.add_patch(badge)
            ax.text(
                x + cell_w - 0.5,
                header_y + header_h - 0.26,
                "NEW",
                ha="center",
                va="center",
                fontsize=7.4,
                fontweight="bold",
                color="#7C2D12",
            )

    for row_index, (row_key, title, subtitle, blurb, accent_color) in enumerate(ROWS):
        y = top_y - row_index * (cell_h + gap_y)
        draw_row_card(
            ax,
            x=row_x,
            y=y,
            width=row_w,
            height=cell_h,
            edgecolor=accent_color,
            title=title,
            subtitle=subtitle,
            body=blurb,
        )

        for col_index, (col_key, _, _, _, _, body_color) in enumerate(REGIMES):
            x = grid_x + col_index * (cell_w + gap_x)
            cell_title, cell_body = CELL_TEXT[(row_key, col_key)]
            draw_card(
                ax,
                x=x,
                y=y,
                width=cell_w,
                height=cell_h,
                edgecolor=accent_color,
                facecolor=body_color,
                title=cell_title,
                body=cell_body,
                title_size=10.1,
                body_size=7.5,
            )

    ax.text(
        0.7,
        2.12,
        "Scenario framing and policy taxonomy are separated so experiment assumptions do not get conflated with the control rules.",
        ha="left",
        va="center",
        fontsize=9.1,
        color="#475467",
    )

    card_y = 0.42
    card_h = 1.36
    card_x = 0.7
    card_gap = 0.2
    card_w = 3.72
    for title, body, edgecolor, facecolor in SUMMARY_CARDS:
        draw_card(
            ax,
            x=card_x,
            y=card_y,
            width=card_w,
            height=card_h,
            edgecolor=edgecolor,
            facecolor=facecolor,
            title=title,
            body=body,
            title_size=9.9,
            body_size=7.2,
            accent_width=0.1,
        )
        card_x += card_w + card_gap

    return save_figure(fig, output_dir, filename)


def main() -> int:
    args = build_parser().parse_args()
    filename = draw_policy_figure(args.out, args.filename)
    print(f"Wrote updated policy figure to {args.out / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())