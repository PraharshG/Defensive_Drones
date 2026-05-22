from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from generate_publication_figures import configure_matplotlib, save_figure


HIGHLIGHTS = [
    ("Attack swarm", "300 attackers\nper experiment", "#0F6CBD", "#EAF2FF"),
    ("Defense sweep", "10-100 defenders\nstep size 10", "#1F9D73", "#ECFDF5"),
    ("Start range", "Attackers start\nabout 5 miles out", "#D97706", "#FFF7ED"),
    ("Primary metric", "Count attackers\nthat still get through", "#B42318", "#FEF3F2"),
]

EXPERIMENT_CHANGES = [
    "Scaled the threat load to\na fixed 300-attacker stress case.",
    "Expanded defender inventory into\na full 10-100 sweep.",
    "Moved the attacker start point farther out\nto about 5 miles from the defended asset.",
    "Added explicit breach accounting so the\nexperiment reports missed attackers.",
]

PARAMETER_KNOBS = [
    "Attacker count /\nattack load",
    "Defender\ncount",
    "Attacker speed",
    "Defender speed",
    "Distance from target",
]

BOTTOM_NOTES = [
    (
        "Worst-case framing",
        "Worst cases are now framed by defender availability\nand attacker stand-off distance.",
        "#111827",
        "#F8FAFC",
    ),
    (
        "Interpretation",
        "Experiment changes only.\nNo policy or assignment-rule changes are included here.",
        "#475467",
        "#F9FAFB",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a single-slide summary of experiment changes."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/publication_figures"),
        help="Directory for PNG/PDF outputs.",
    )
    parser.add_argument(
        "--filename",
        default="13_experiment_changes_summary_slide",
        help="Base filename for the rendered slide.",
    )
    return parser


def add_card(
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
    title_size: float = 14.0,
    body_size: float = 11.2,
    accent_width: float = 0.0,
) -> None:
    card = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.5,
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
    text_x = x + accent_width + 0.18
    ax.text(
        text_x,
        y + height - 0.2,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        text_x,
        y + 0.22,
        body,
        ha="left",
        va="bottom",
        fontsize=body_size,
        color="#374151",
    )


def draw_bullet_list(
    ax: plt.Axes,
    *,
    x: float,
    y_top: float,
    lines: list[str],
    bullet_color: str,
    text_size: float,
    line_gap: float,
) -> None:
    y = y_top
    for line in lines:
        ax.text(x, y, "•", ha="left", va="top", fontsize=text_size + 3.0, color=bullet_color)
        ax.text(x + 0.22, y, line, ha="left", va="top", fontsize=text_size, color="#374151")
        y -= line_gap


def draw_knob_chip(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    text: str,
    edgecolor: str,
    facecolor: str,
) -> None:
    chip = patches.FancyBboxPatch(
        (x, y),
        width,
        0.54,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(chip)
    ax.text(
        x + 0.16,
        y + 0.27,
        text,
        ha="left",
        va="center",
        fontsize=9.0,
        color="#1F2937",
    )


def draw_slide(output_dir: Path, filename: str) -> str:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0.0, 13.333)
    ax.set_ylim(0.0, 7.5)
    ax.axis("off")

    ax.add_patch(patches.Circle((1.1, 6.8), 1.3, color="#EAF2FF", alpha=0.95, linewidth=0))
    ax.add_patch(patches.Circle((12.3, 1.0), 1.5, color="#ECFDF5", alpha=0.95, linewidth=0))
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.55, 0.45),
            12.2,
            6.45,
            boxstyle="round,pad=0.03,rounding_size=0.26",
            linewidth=1.0,
            edgecolor="#E5E7EB",
            facecolor="#FFFFFF",
        )
    )

    fig.text(
        0.07,
        0.93,
        "Experiment Changes Summary",
        ha="left",
        va="top",
        fontsize=24,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.885,
        "Scope: experiment setup only. No policy or solution logic changes are summarized on this slide.",
        ha="left",
        va="top",
        fontsize=11.5,
        color="#475467",
    )

    card_x = 0.85
    card_y = 5.65
    card_w = 2.85
    card_h = 1.02
    card_gap = 0.16
    for title, body, edgecolor, facecolor in HIGHLIGHTS:
        add_card(
            ax,
            x=card_x,
            y=card_y,
            width=card_w,
            height=card_h,
            edgecolor=edgecolor,
            facecolor=facecolor,
            title=title,
            body=body,
            title_size=11.4,
            body_size=9.1,
            accent_width=0.1,
        )
        card_x += card_w + card_gap

    add_card(
        ax,
        x=0.85,
        y=2.45,
        width=6.05,
        height=2.8,
        edgecolor="#111827",
        facecolor="#F8FAFC",
        title="What Changed in the Experiment",
        body="",
        title_size=15.5,
        body_size=11.0,
        accent_width=0.12,
    )
    draw_bullet_list(
        ax,
        x=1.15,
        y_top=4.65,
        lines=EXPERIMENT_CHANGES,
        bullet_color="#0F6CBD",
        text_size=10.8,
        line_gap=0.55,
    )

    add_card(
        ax,
        x=7.15,
        y=2.45,
        width=4.95,
        height=2.8,
        edgecolor="#1D4ED8",
        facecolor="#F8FBFF",
        title="Knobs and Readouts",
        body="",
        title_size=14.6,
        body_size=10.1,
        accent_width=0.12,
    )

    ax.text(
        7.45,
        4.32,
        "Sweep along these experiment knobs:",
        ha="left",
        va="top",
        fontsize=10.1,
        color="#475467",
    )

    chip_x = 7.45
    chip_y = 3.72
    chip_w = 2.15
    chip_gap_x = 0.2
    chip_gap_y = 0.16
    for index, text in enumerate(PARAMETER_KNOBS):
        row = index // 2
        col = index % 2
        x = chip_x + col * (chip_w + chip_gap_x)
        y = chip_y - row * (0.54 + chip_gap_y)
        if index == 4:
            x = chip_x
            y = chip_y - 2 * (0.54 + chip_gap_y)
            chip_w = 4.5
        draw_knob_chip(
            ax,
            x=x,
            y=y,
            width=chip_w,
            text=text,
            edgecolor="#BFDBFE",
            facecolor="#EFF6FF",
        )
        if index == 4:
            chip_w = 2.15

    add_card(
        ax,
        x=0.85,
        y=0.8,
        width=5.85,
        height=1.18,
        edgecolor=BOTTOM_NOTES[0][2],
        facecolor=BOTTOM_NOTES[0][3],
        title=BOTTOM_NOTES[0][0],
        body=BOTTOM_NOTES[0][1],
        title_size=11.8,
        body_size=9.3,
        accent_width=0.1,
    )
    add_card(
        ax,
        x=6.95,
        y=0.8,
        width=5.15,
        height=1.18,
        edgecolor=BOTTOM_NOTES[1][2],
        facecolor=BOTTOM_NOTES[1][3],
        title=BOTTOM_NOTES[1][0],
        body=BOTTOM_NOTES[1][1],
        title_size=11.8,
        body_size=9.1,
        accent_width=0.1,
    )

    return save_figure(fig, output_dir, filename)


def main() -> int:
    args = build_parser().parse_args()
    filename = draw_slide(args.out, args.filename)
    print(f"Wrote experiment summary slide to {args.out / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())