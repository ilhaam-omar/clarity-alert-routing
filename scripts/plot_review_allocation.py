"""Plot matched review-recipient overlap from aggregate comparison results.

Example from the repository root:
    python scripts/plot_review_allocation.py \
        --input results/matched/paired_comparisons.csv \
        --output-dir figures

Only aggregate counts are read. No patient-level records are required.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


SCENARIOS = [
    ("low_capacity", "Low"),
    ("medium_capacity", "Medium"),
    ("high_capacity", "High"),
    ("unconstrained_reference", "No-scarcity reference"),
]
CONTRAST = "ranking_with_clarity_routes"
LEFT_POLICY = "B_clarity_severity"
RIGHT_POLICY = "D_clarity_regret"


def load_counts(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["contrast"] == CONTRAST]
    by_scenario = {row["scenario"]: row for row in rows}
    if len(rows) != len(SCENARIOS) or set(by_scenario) != {s for s, _ in SCENARIOS}:
        raise ValueError("Expected exactly one B-D comparison for each of four scenarios.")

    selected = []
    for scenario, label in SCENARIOS:
        row = by_scenario[scenario]
        if row["left_policy"] != LEFT_POLICY or row["right_policy"] != RIGHT_POLICY:
            raise ValueError("The selected contrast must compare policies B and D.")
        left_total = int(row["left_review_recipients"])
        right_total = int(row["right_review_recipients"])
        common = int(row["common_review_recipients"])
        left_only = int(row["left_only_review_recipients"])
        right_only = int(row["right_only_review_recipients"])
        if left_total != right_total or left_only != right_only:
            raise ValueError("Matched policies must have equal review counts and unique counts.")
        if common + left_only != left_total or min(common, left_only) < 0:
            raise ValueError("Shared and policy-specific counts must partition each review set.")
        selected.append({
            "scenario": scenario,
            "scenario_label": label,
            "left_policy": LEFT_POLICY,
            "right_policy": RIGHT_POLICY,
            "review_recipients_per_policy": left_total,
            "common_review_recipients": common,
            "recipients_unique_to_each_policy": left_only,
            "replacement_percent_per_policy": 100 * left_only / left_total,
        })
    return selected


def plot_counts(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#76818A",
        "axes.linewidth": 0.65,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.1, 3.8), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[5.2, 1], wspace=0.02)
    ax = fig.add_subplot(grid[0, 0])
    percentage_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    common_color = "#C4D0D8"
    unique_color = "#287F79"
    text_color = "#203443"

    for index, row in enumerate(rows):
        common = row["common_review_recipients"]
        unique = row["recipients_unique_to_each_policy"]
        ax.barh(index, common, height=0.58, color=common_color, edgecolor="white", linewidth=0.6)
        if unique:
            ax.barh(index, unique, left=common, height=0.58,
                    color=unique_color, edgecolor="white", linewidth=0.6)
        ax.text(common / 2, index, f"{common:,}", va="center", ha="center",
                color=text_color, fontsize=9)
        if unique:
            ax.text(common + unique / 2, index, f"{unique:,}", va="center", ha="center",
                    color="white", fontsize=9)
        percentage_ax.text(0.5, index, f"{row['replacement_percent_per_policy']:.1f}%",
                           va="center", ha="center", color=text_color, fontsize=10,
                           fontweight="semibold")

    ax.set_yticks(range(len(rows)), [
        f"{row['scenario_label']}\n(n = {row['review_recipients_per_policy']:,} per policy)"
        for row in rows
    ])
    ax.set_ylim(3.65, -0.75)
    ax.set_xlim(0, 4100)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("Reviewed bundles per policy", labelpad=8)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.tick_params(axis="x", color="#76818A", length=3)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E4E9ED", linewidth=0.6)
    percentage_ax.set_xlim(0, 1)
    percentage_ax.set_axis_off()
    percentage_ax.text(0.5, -0.73, "Replaced\nwithin set", va="bottom", ha="center",
                       color=text_color, fontsize=8.5)
    fig.legend(handles=[
        Patch(facecolor=common_color, label="Common to both policies"),
        Patch(facecolor=unique_color, label="Specific to each policy"),
    ], loc="upper center", bbox_to_anchor=(0.61, 0.99), ncol=2,
       frameon=False, columnspacing=1.7, handlelength=1.4, fontsize=8.8)
    fig.subplots_adjust(left=0.245, right=0.995, top=0.81, bottom=0.20)
    fig.text(0.245, 0.025, "Each bar represents one policy's review set, not the union of both sets.",
             ha="left", va="bottom", fontsize=8, color="#495B68")

    stem = output_dir / "figure_review_recipient_overlap"
    fig.savefig(stem.with_suffix(".pdf"), metadata={
        "Title": "Review recipient overlap under matched capacity",
        "Subject": "Conditional comparison of severity and regret-based queue priorities",
    })
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    with (output_dir / "figure_review_recipient_overlap_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Aggregate paired_comparisons.csv from the matched experiment.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    plot_counts(load_counts(args.input), args.output_dir)


if __name__ == "__main__":
    main()
