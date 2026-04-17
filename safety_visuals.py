from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

COLORS = {
    "ACT Campus": "#7A1E2C",
    "Housing": "#111111",
    "Thessaloniki": "#7A7A7A",
    "Travel in Greece": "#C55C6C",
}


def create_safety_line_chart(summary_csv, output_png, min_n=20):
    summary = pd.read_csv(summary_csv)
    summary = summary[summary["n"] >= min_n].copy()

    summary.loc[
        (summary["Year_num"] == 2023) & (summary["category"] == "Housing"),
        "positive_pct"
    ] = np.nan

    pivot = summary.pivot(index="Year_num", columns="category", values="positive_pct").sort_index()

    plt.figure(figsize=(9.5, 5.3))

    for category in ["ACT Campus", "Housing", "Thessaloniki", "Travel in Greece"]:
        if category in pivot.columns:
            plt.plot(
                pivot.index,
                pivot[category] * 100,
                linewidth=4,
                marker="o",
                markersize=8,
                color=COLORS[category],
                label=category,
            )

    plt.title("Student Safety Perception Over Time", fontsize=18, weight="bold")
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("% Agree or Strongly Agree", fontsize=12)
    plt.ylim(88, 101)
    plt.grid(axis="y", alpha=0.2, linestyle="--")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    output_png = Path(output_png)
    plt.savefig(output_png, dpi=320, bbox_inches="tight")
    plt.close()
    return output_png


def create_safety_heatmap(summary_csv, output_png, min_n=20):
    summary = pd.read_csv(summary_csv)
    summary = summary[summary["n"] >= min_n].copy()

    heat_df = summary.pivot(index="Year_num", columns="category", values="positive_pct").sort_index()
    heat_df = heat_df.loc[:, ["ACT Campus", "Housing", "Thessaloniki", "Travel in Greece"]]

    cmap = LinearSegmentedColormap.from_list(
        "burgundy_scale",
        ["#F7E3E7", "#E2A6B2", "#C05B70", "#7A1E2C"],
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(heat_df.T.values * 100, aspect="auto", cmap=cmap, vmin=88, vmax=100)

    ax.set_title("Safety Positivity by Year", fontsize=22, weight="bold", pad=22)

    fig.text(
        0.5,
        0.88,
        "Percent of students reporting feeling safe or very safe in each environment\n"
        "(years filtered to at least 20 responses per item)",
        ha="center",
        fontsize=12,
        color="#444444",
    )

    ax.set_xticks(np.arange(len(heat_df.index)))
    ax.set_xticklabels(heat_df.index.astype(int), fontsize=11)
    ax.set_yticks(np.arange(len(heat_df.columns)))
    ax.set_yticklabels(heat_df.columns, fontsize=13)

    for row in range(heat_df.shape[1]):
        for col in range(heat_df.shape[0]):
            value = heat_df.iloc[col, row] * 100
            text_color = "white" if value >= 96 else "black"
            ax.text(
                col,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                weight="bold",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("% Positive Responses", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.9])

    output_png = Path(output_png)
    plt.savefig(output_png, dpi=320, bbox_inches="tight")
    plt.close()
    return output_png
