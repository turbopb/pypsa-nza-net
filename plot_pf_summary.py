#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_pf_summary.py

Cross-month DC power flow summary figure for SI transmission network.

Produces a two-panel figure:
  Top:    Heatmap -- top-N lines x 24 months, colour = max loading (%)
  Bottom: S-37 monthly max loading bar chart with seasonal annotation

Reads line_loading.csv from each month's verification_pf output folder.

Usage
-----
    python plot_pf_summary.py [--top-n 20] [--out <path>]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
YEARS = [2024, 2025]

# S-37 is ISL-KIK 220 kV -- the validated binding winter constraint
HIGHLIGHT_LINE = "S-37"


def load_all_results(workspace: Path) -> pd.DataFrame:
    """
    Load line_loading.csv for all 24 months.
    Returns DataFrame: index=line_id, columns=MultiIndex(year, month_abbr).
    Only max_loading_pct is loaded.
    """
    frames = {}
    for year in YEARS:
        for m_idx, m_name in enumerate(MONTHS, start=1):
            month_label = f"{m_name}_{year}"
            csv_path = (workspace / "outputs" / "verification_pf"
                        / str(year) / month_label / "line_loading.csv")
            if not csv_path.exists():
                log.warning("  Missing: %s", csv_path)
                continue
            df = pd.read_csv(str(csv_path), index_col=0)
            if "max_loading_pct" not in df.columns:
                log.warning("  No max_loading_pct in %s", csv_path)
                continue
            frames[(year, m_name)] = df["max_loading_pct"]
            log.info("  Loaded %s: %d lines", month_label, len(df))

    if not frames:
        log.error("No data loaded.")
        sys.exit(1)

    result = pd.DataFrame(frames)
    result.columns = pd.MultiIndex.from_tuples(result.columns,
                                               names=["year", "month"])
    log.info("Loaded %d months x %d lines", len(result.columns), len(result))
    return result


def make_figure(data: pd.DataFrame, top_n: int, out_path: Path):
    """
    Two-panel figure:
      Top:    Heatmap of top-N most loaded lines across all months
      Bottom: S-37 monthly max loading bars
    """
    # --- Select top-N lines by peak loading across all months ---
    peak = data.max(axis=1).sort_values(ascending=False)
    top_lines = peak.head(top_n).index.tolist()
    heatmap_data = data.loc[top_lines]          # (top_n, 24)
    heatmap_arr  = heatmap_data.values.astype(float)

    # Column labels: Jan 2024 ... Dec 2025
    col_labels = []
    for year in YEARS:
        for m in MONTH_LABELS:
            col_labels.append(f"{m}\n{year}")

    # --- Custom colormap: white → steel blue → orange → red ---
    cmap = LinearSegmentedColormap.from_list(
        "loading",
        [(0.00, "#f7fbff"),
         (0.50, "#4292c6"),
         (0.75, "#fd8d3c"),
         (1.00, "#d73027")],
    )

    # --- Figure layout ---
    fig = plt.figure(figsize=(16, 9))
    gs  = fig.add_gridspec(
        2, 1,
        height_ratios=[top_n * 0.55, 3.5],
        hspace=0.38,
    )
    ax_heat = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    # ----------------------------------------------------------------
    # TOP PANEL: heatmap
    # ----------------------------------------------------------------
    im = ax_heat.imshow(
        heatmap_arr,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=110,
        interpolation="nearest",
    )

    # Annotate cells with loading value
    for r in range(len(top_lines)):
        for c in range(len(col_labels)):
            val = heatmap_arr[r, c]
            if np.isnan(val):
                continue
            txt_col = "white" if val > 65 else "#333333"
            ax_heat.text(c, r, f"{val:.0f}",
                         ha="center", va="center",
                         fontsize=6.5, color=txt_col, fontweight="bold")

    # Draw year divider
    ax_heat.axvline(11.5, color="#555555", linewidth=1.2, linestyle="--")

    ax_heat.set_xticks(range(len(col_labels)))
    ax_heat.set_xticklabels(col_labels, fontsize=7.5)
    ax_heat.set_yticks(range(len(top_lines)))
    ax_heat.set_yticklabels(top_lines, fontsize=8, fontfamily="monospace")
    ax_heat.set_title(
        f"SI Transmission Line Loading — Top {top_n} Lines (Max % of Rated Capacity)",
        fontsize=11, fontweight="bold", pad=8,
    )

    # Year labels above divider
    ax_heat.text(5.5,  -1.7, "2024", ha="center", fontsize=9,
                 fontweight="bold", color="#333333",
                 transform=ax_heat.get_xaxis_transform())
    ax_heat.text(17.5, -1.7, "2025", ha="center", fontsize=9,
                 fontweight="bold", color="#333333",
                 transform=ax_heat.get_xaxis_transform())

    # Colourbar
    cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical",
                        fraction=0.015, pad=0.01)
    cbar.set_label("Max loading (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Highlight 100% on colourbar
    cbar.ax.axhline(100 / 110, color="red", linewidth=1.2, linestyle="--")

    # ----------------------------------------------------------------
    # BOTTOM PANEL: S-37 bar chart
    # ----------------------------------------------------------------
    if HIGHLIGHT_LINE in data.index:
        s37 = data.loc[HIGHLIGHT_LINE]
    else:
        log.warning("  %s not found in data -- skipping bar panel", HIGHLIGHT_LINE)
        s37 = pd.Series(dtype=float)

    x_pos = np.arange(len(col_labels))
    bar_vals = s37.values.astype(float) if len(s37) else np.full(len(col_labels), np.nan)

    bar_colors = []
    for v in bar_vals:
        if np.isnan(v):
            bar_colors.append("#cccccc")
        elif v >= 100:
            bar_colors.append("#d73027")
        elif v >= 75:
            bar_colors.append("#fd8d3c")
        else:
            bar_colors.append("#4292c6")

    bars = ax_bar.bar(x_pos, bar_vals, color=bar_colors,
                      edgecolor="white", linewidth=0.4, width=0.75)

    # 100% and 75% threshold lines
    ax_bar.axhline(100, color="#d73027", linestyle="--",
                   linewidth=1.2, label="100% rated")
    ax_bar.axhline(75,  color="#fd8d3c", linestyle=":",
                   linewidth=1.0, label="75% threshold")

    # Year divider
    ax_bar.axvline(11.5, color="#555555", linewidth=1.2, linestyle="--")

    # Annotate peak bar
    peak_idx = int(np.nanargmax(bar_vals))
    ax_bar.annotate(
        f"{bar_vals[peak_idx]:.1f}%",
        xy=(x_pos[peak_idx], bar_vals[peak_idx]),
        xytext=(0, 5), textcoords="offset points",
        ha="center", fontsize=8, fontweight="bold", color="#d73027",
    )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(col_labels, fontsize=7.5)
    ax_bar.set_ylabel("Max loading (% rated)", fontsize=9)
    ax_bar.set_ylim(0, 115)
    ax_bar.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax_bar.set_title(
        f"S-37 (ISL–KIK 220 kV) — Monthly Maximum Loading",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    ax_bar.spines[["top", "right"]].set_visible(False)

    # Year labels
    ax_bar.text(5.5,  108, "2024", ha="center", fontsize=9,
                fontweight="bold", color="#333333")
    ax_bar.text(17.5, 108, "2025", ha="center", fontsize=9,
                fontweight="bold", color="#333333")

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-month SI power flow summary figure")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of most loaded lines in heatmap (default 20)")
    parser.add_argument("--out",   type=str, default=None,
                        help="Output PNG path (default: workspace/outputs/verification_pf/pf_summary.png)")
    args = parser.parse_args()

    _root = os.environ.get("PYPSA_NZA_ROOT", "")
    if not _root:
        log.error("PYPSA_NZA_ROOT not set.")
        sys.exit(1)

    workspace = Path(_root) / "pypsa_nza_workspace"
    out_path  = Path(args.out) if args.out else (
        workspace / "outputs" / "verification_pf" / "pf_summary.png"
    )

    log.info("Loading 24-month power flow results...")
    data = load_all_results(workspace)

    log.info("Generating figure (top %d lines)...", args.top_n)
    make_figure(data, top_n=args.top_n, out_path=out_path)


if __name__ == "__main__":
    main()
