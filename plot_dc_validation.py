#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_dc_validation.py

Publication-quality plots for the DC power flow validation chapter.

Generates:
1. SI network map with lines colour-coded by mean loading (Jan 2024)
2. Line loading comparison bar chart: Jan vs Jul 2024
3. Monthly mean generation, HVDC, and load for 2024

Usage
-----
    python plot_dc_validation.py

Outputs written to:
    {workspace}/outputs/figures/validation/
        si_network_loading_jan2024.png
        si_line_loading_comparison.png
        si_monthly_balance_2024.png

Author: pypsa-nza-net
"""

import os
import sys
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

_root = os.environ.get("PYPSA_NZA_ROOT", "")
if not _root:
    log.error("PYPSA_NZA_ROOT not set")
    sys.exit(1)

workspace  = Path(_root) / "pypsa_nza_workspace"
manual_dir = workspace / "data" / "manual"
verif_dir  = workspace / "outputs" / "verification"
fig_dir    = workspace / "outputs" / "figures" / "validation"
fig_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

def load_nodes():
    """Load node coordinates from nodes.csv (WGS84)."""
    df = pd.read_csv(str(manual_dir / "nodes.csv"), dtype=str)
    df["site"] = df["site"].str.strip().str.upper()
    df["lat"]  = pd.to_numeric(df["lat"],  errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    return df.set_index("site")

def load_lines():
    """Load SI line data."""
    df = pd.read_csv(str(manual_dir / "lines_data_si.csv"), dtype=str)
    df.columns = df.columns.str.strip()
    # Identify bus0/bus1 columns
    for c in df.columns:
        log.info("  col: %s", c)
    return df

def load_loading(year, month):
    """Load dc_line_loading.csv for a given month."""
    month_label = f"{['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'][month-1]}_{year}"
    path = verif_dir / str(year) / month_label / "dc_line_loading.csv"
    if not path.exists():
        log.warning("Not found: %s", path)
        return None
    df = pd.read_csv(str(path), index_col=0)
    return df

def load_flows(year, month):
    """Load dc_line_flows.csv for a given month."""
    month_label = f"{['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'][month-1]}_{year}"
    path = verif_dir / str(year) / month_label / "dc_line_flows.csv"
    if not path.exists():
        return None
    df = pd.read_csv(str(path), index_col=0, parse_dates=True)
    return df

# =============================================================================
# FIGURE 1: NETWORK MAP WITH LINE LOADING
# =============================================================================

def plot_network_map(year=2024, month=1):
    """
    SI network map with lines coloured by mean loading percentage.
    Publication quality with clear annotations.
    """
    log.info("Figure 1: Network map (month=%d)", month)

    nodes    = load_nodes()
    lines_df = pd.read_csv(str(manual_dir / "lines_data_si.csv"), dtype=str)
    loading  = load_loading(year, month)

    if loading is None:
        log.error("No loading data for %d/%d", year, month)
        return

    def get_coords(site):
        site = str(site).strip().upper()
        if site in nodes.index:
            row = nodes.loc[site]
            if pd.notna(row["lat"]) and pd.notna(row["long"]):
                return float(row["lat"]), float(row["long"])
        base = site.rsplit("_", 1)[0]
        if base in nodes.index:
            row = nodes.loc[base]
            if pd.notna(row["lat"]) and pd.notna(row["long"]):
                return float(row["lat"]), float(row["long"])
        return None

    cmap = matplotlib.colormaps["RdYlGn_r"]
    norm = mcolors.Normalize(vmin=0, vmax=100)

    fig, ax = plt.subplots(figsize=(10, 16), facecolor="#f5f4f0")
    ax.set_facecolor("#eaeef2")

    plotted_buses = set()

    # --- Draw lines ---
    for _, row in lines_df.iterrows():
        cols  = lines_df.columns.tolist()
        line_id = str(row.iloc[0]).strip()
        # Find bus columns by position (lines_data_si: id,name,desc,bus0,bus1,...)
        b0 = str(row.iloc[3]).strip().upper()
        b1 = str(row.iloc[4]).strip().upper()

        c0 = get_coords(b0)
        c1 = get_coords(b1)
        if c0 is None or c1 is None:
            continue

        plotted_buses.add(b0)
        plotted_buses.add(b1)

        pct = float(loading.loc[line_id, "mean_loading_pct"]) \
              if line_id in loading.index else 0.0
        mx  = float(loading.loc[line_id, "max_loading_pct"]) \
              if line_id in loading.index else 0.0

        color = cmap(norm(pct))
        # Line thickness dramatically scaled: thin=lightly loaded, thick=heavily loaded
        lw    = 0.8 + (pct / 100) ** 1.5 * 8.0

        ax.plot([c0[1], c1[1]], [c0[0], c1[0]],
                color=color, linewidth=lw,
                solid_capstyle="round", zorder=2,
                alpha=0.9)

        # Label lines with mean loading > 25%
        if pct > 25:
            mid_lat = (c0[0] + c1[0]) / 2
            mid_lon = (c0[1] + c1[1]) / 2
            ax.annotate(
                f"{line_id}  {pct:.0f}% mean\n(max {mx:.0f}%)",
                xy=(mid_lon, mid_lat),
                xytext=(mid_lon + 0.12, mid_lat + 0.04),
                fontsize=6.5, color="#1a1a1a",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-",
                                color="#555555",
                                lw=0.7),
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white",
                          edgecolor="#aaaaaa",
                          alpha=0.92),
                zorder=6
            )

    # --- Bus categories ---
    gen_buses  = {"MAN","BEN","CYD","ROX","OHA","OHB","OHC","AVI",
                  "TKB","ARG","BWK","GOR","WTK","COL","TKA","NMA","STK"}
    load_buses = {"ISL","TWI","INV","TIM","BLN","HWB","NWD","TWZ",
                  "CML","KIK","ABY","ORO"}

    # Manual label offsets (lon, lat) for key buses to avoid overlap
    label_offsets = {
        "MAN":  (-0.55,  0.02), "BEN":  (-0.40,  0.05),
        "CYD":  (-0.35, -0.06), "ROX":  ( 0.07,  0.04),
        "OHA":  (-0.42,  0.04), "OHB":  (-0.42, -0.04),
        "OHC":  (-0.42, -0.10), "AVI":  ( 0.07,  0.04),
        "TKB":  (-0.35,  0.05), "ARG":  ( 0.07,  0.02),
        "WTK":  ( 0.07, -0.05), "GOR":  ( 0.07,  0.03),
        "ISL":  ( 0.07,  0.04), "TWI":  ( 0.07,  0.02),
        "INV":  (-0.38, -0.05), "TIM":  ( 0.08,  0.02),
        "BLN":  ( 0.08,  0.02), "NWD":  ( 0.08, -0.05),
        "HWB":  ( 0.08,  0.02), "KIK":  ( 0.08,  0.02),
        "STK":  ( 0.08,  0.02), "NMA":  (-0.38,  0.02),
        "TWZ":  ( 0.08,  0.02), "CML":  (-0.35,  0.02),
        "ABY":  ( 0.08,  0.02), "ORO":  (-0.35,  0.02),
        "TKA":  ( 0.08,  0.02), "MCH":  ( 0.08,  0.02),
    }

    key_labels = {"MAN","BEN","CYD","ROX","OHA","OHB","OHC","AVI","TKB",
                  "ISL","TWI","INV","TIM","BLN","HWB","NWD","KIK","STK",
                  "NMA","WTK","CML","TWZ","ARG","GOR","TKA","ABY","ORO",
                  "MCH","NSY"}

    for bus in plotted_buses:
        c = get_coords(bus)
        if c is None:
            continue
        base = bus.rsplit("_", 1)[0]

        if base in gen_buses:
            ms, mk, mc, zord = 8, "o", "#1565C0", 4
        elif base in load_buses:
            ms, mk, mc, zord = 7, "s", "#B71C1C", 4
        else:
            ms, mk, mc, zord = 4, "o", "#546E7A", 3

        ax.plot(c[1], c[0], mk, color=mc,
                markersize=ms, zorder=zord,
                markeredgecolor="white",
                markeredgewidth=1.0)

        if base in key_labels:
            off_lon, off_lat = label_offsets.get(base, (0.07, 0.02))
            ax.text(c[1] + off_lon, c[0] + off_lat,
                    base,
                    fontsize=7.5,
                    color="#111111",
                    fontweight="bold",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white",
                              edgecolor="none",
                              alpha=0.8),
                    zorder=7)

    # --- Colourbar ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.3, pad=0.02,
                        aspect=18, location="right")
    cbar.set_label("Mean line loading (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0, 25, 50, 75, 100])

    # --- Legend ---
    legend_elements = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#1565C0", markersize=9,
               markeredgecolor="white",
               label="Generation bus"),
        Line2D([0],[0], marker="s", color="w",
               markerfacecolor="#B71C1C", markersize=8,
               markeredgecolor="white",
               label="Load / substation bus"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#546E7A", markersize=6,
               markeredgecolor="white",
               label="Junction / transformer bus"),
        Line2D([0],[0], color=cmap(norm(0)),
               linewidth=2.5, label="Lightly loaded line"),
        Line2D([0],[0], color=cmap(norm(50)),
               linewidth=3.5, label="Moderately loaded line"),
        Line2D([0],[0], color=cmap(norm(90)),
               linewidth=5.0, label="Heavily loaded line"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=8, framealpha=0.95,
              edgecolor="#bbbbbb", borderpad=0.8)

    month_name = ["January","February","March","April","May","June",
                  "July","August","September","October","November",
                  "December"][month-1]
    ax.set_title(
        f"South Island Transmission Network\n"
        f"DC Power Flow Validation — {month_name} {year}\n"
        f"Lines coloured and weighted by mean loading (% of rated capacity)",
        fontsize=10, pad=14, fontweight="bold"
    )
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°S)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.5, color="#888888")
    ax.set_xlim(166.2, 174.2)
    ax.set_ylim(-47.2, -40.4)

    fig.tight_layout()
    out = fig_dir / f"si_network_loading_{month_name.lower()}{year}.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("  Saved: %s", out)
    return out


# =============================================================================
# FIGURE 2: LINE LOADING COMPARISON JAN vs JUL
# =============================================================================

def plot_loading_comparison():
    """
    Side-by-side bar chart of line loading for Jan and Jul 2024.
    Shows top 20 lines by max loading in either month.
    """
    log.info("Figure 2: Line loading comparison Jan vs Jul")

    jan = load_loading(2024, 1)
    jul = load_loading(2024, 7)

    if jan is None or jul is None:
        log.error("Missing loading data")
        return

    # Union of lines, sorted by max of either month
    all_lines = sorted(set(jan.index) | set(jul.index))
    max_either = []
    for ln in all_lines:
        j = float(jan.loc[ln, "max_loading_pct"]) if ln in jan.index else 0
        u = float(jul.loc[ln, "max_loading_pct"]) if ln in jul.index else 0
        max_either.append(max(j, u))

    # Take top 20
    top_idx = sorted(range(len(all_lines)),
                     key=lambda i: max_either[i], reverse=True)[:20]
    top_lines = [all_lines[i] for i in top_idx]

    jan_mean = [float(jan.loc[ln, "mean_loading_pct"])
                if ln in jan.index else 0 for ln in top_lines]
    jan_max  = [float(jan.loc[ln, "max_loading_pct"])
                if ln in jan.index else 0 for ln in top_lines]
    jul_mean = [float(jul.loc[ln, "mean_loading_pct"])
                if ln in jul.index else 0 for ln in top_lines]
    jul_max  = [float(jul.loc[ln, "max_loading_pct"])
                if ln in jul.index else 0 for ln in top_lines]

    y   = np.arange(len(top_lines))
    h   = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                              facecolor="white", sharey=True)

    colours = {"jan_mean": "#4FC3F7", "jan_max": "#0277BD",
               "jul_mean": "#FFCC80", "jul_max": "#E65100"}

    for ax, mean_vals, max_vals, label, c_mean, c_max in [
        (axes[0], jan_mean, jan_max, "January 2024",
         colours["jan_mean"], colours["jan_max"]),
        (axes[1], jul_mean, jul_max, "July 2024",
         colours["jul_mean"], colours["jul_max"]),
    ]:
        ax.barh(y + h/2, max_vals,  h, color=c_max,
                label="Maximum", alpha=0.9)
        ax.barh(y - h/2, mean_vals, h, color=c_mean,
                label="Mean", alpha=0.9)
        ax.axvline(100, color="red", linestyle="--",
                   linewidth=1.2, label="Rated capacity")
        ax.axvline(75,  color="orange", linestyle=":",
                   linewidth=1.0, label="75% threshold")
        ax.set_xlabel("Loading (% of rated capacity)", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_xlim(0, max(max(jan_max + jul_max) * 1.1, 105))
        ax.tick_params(labelsize=8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(top_lines, fontsize=8)

    fig.suptitle(
        "South Island Transmission Line Loading — DC Power Flow Validation\n"
        "Top 20 lines by maximum loading, 2024",
        fontsize=11, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    out = fig_dir / "si_line_loading_comparison.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", out)
    return out


# =============================================================================
# FIGURE 3: MONTHLY POWER BALANCE 2024
# =============================================================================

def plot_monthly_balance():
    """
    Monthly mean SI generation (stacked by site), HVDC import,
    and SI load for all 12 months of 2024.
    """
    log.info("Figure 3: Monthly power balance 2024")

    annual_dir = workspace / "data" / "processed" / "annual"
    hvdc_base  = workspace / "data" / "processed" / "2024" / "hvdc" / "dc_MW"
    nodes      = load_nodes()

    # SI GIP sites
    si_sites = set(nodes[nodes.get("island", pd.Series()) == "SI"].index
                   if "island" in nodes.columns else [])

    df_imp = pd.read_csv(str(annual_dir / "2024_import_sites_all.csv"),
                         index_col=0, parse_dates=True)
    df_exp = pd.read_csv(str(annual_dir / "2024_export_sites_all.csv"),
                         index_col=0, parse_dates=True)

    # Filter to SI
    si_imp_cols = [c for c in df_imp.columns
                   if c.strip().upper() in si_sites]
    si_exp_cols = [c for c in df_exp.columns
                   if c.strip().upper() in si_sites]
    df_imp_si = df_imp[si_imp_cols]
    df_exp_si = df_exp[si_exp_cols]

    months     = range(1, 13)
    month_abbr = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

    gen_means  = []
    load_means = []
    hvdc_means = []

    # Key generator sites for stacking
    gen_sites = ["MAN","BEN","CYD","OHA","OHB","OHC","ROX","AVI","TKB",
                 "ARG","BWK","WTK","GOR","COL","TKA","NMA","STK","MCH","ABY"]
    site_colours = {
        "MAN": "#1565C0", "BEN": "#1976D2", "CYD": "#1E88E5",
        "OHA": "#42A5F5", "OHB": "#64B5F6", "OHC": "#90CAF9",
        "ROX": "#0D47A1", "AVI": "#2196F3", "TKB": "#82B1FF",
        "WTK": "#43A047", "GOR": "#66BB6A", "COL": "#A5D6A7",
        "TKA": "#388E3C", "NMA": "#81C784", "STK": "#C8E6C9",
        "MCH": "#A5D6A7", "ABY": "#DCEDC8",
        "ARG": "#B0BEC5", "BWK": "#CFD8DC",
    }

    monthly_gen  = {s: [] for s in gen_sites}

    for m in months:
        mask = df_imp_si.index.month == m
        for s in gen_sites:
            if s in df_imp_si.columns:
                monthly_gen[s].append(float(df_imp_si.loc[mask, s].mean()))
            else:
                monthly_gen[s].append(0.0)

        # Load
        mask_exp = df_exp_si.index.month == m
        load_means.append(float(df_exp_si.loc[mask_exp].sum(axis=1).mean()))

        # HVDC
        hvdc_path = hvdc_base / f"2024{m:02d}_dc_MW.csv"
        if hvdc_path.exists():
            dh = pd.read_csv(str(hvdc_path), index_col=0, parse_dates=True)
            if "BEN2201" in dh.columns:
                hvdc_means.append(float(dh["BEN2201"].mean()))
            else:
                hvdc_means.append(0.0)
        else:
            hvdc_means.append(0.0)

    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")

    # Stacked generation bars
    bottom = np.zeros(12)
    for s in gen_sites:
        vals = np.array(monthly_gen[s])
        if vals.sum() > 0:
            ax.bar(x, vals, bottom=bottom,
                   color=site_colours.get(s, "#90A4AE"),
                   label=s, alpha=0.9, width=0.6)
            bottom += vals

    # HVDC as hatched overlay
    ax.bar(x, np.array(hvdc_means), bottom=bottom,
           color="#FF8F00", alpha=0.7, width=0.6,
           hatch="///", label="HVDC import (BEN2201)")

    # Load line
    ax.plot(x, load_means, "ko-", linewidth=2,
            markersize=6, label="SI load (GXPs)", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(month_abbr, fontsize=9)
    ax.set_ylabel("Mean power (MW)", fontsize=10)
    ax.set_title(
        "South Island Power Balance — Monthly Means, 2024\n"
        "Stacked SI generation + HVDC import vs SI load",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.legend(ncol=4, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.0),
              framealpha=0.95, edgecolor="#cccccc")
    ax.grid(True, axis="y", alpha=0.3)
    # Extend y-axis by 35% above max bar to give legend room
    y_max = max(float(bottom.max()), max(load_means))
    ax.set_ylim(0, y_max * 1.38)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = fig_dir / "si_monthly_balance_2024.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", out)
    return out


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    log.info("Generating DC power flow validation figures...")
    log.info("Output: %s", fig_dir)

    f1a = plot_network_map(2024, 1)   # January
    f1b = plot_network_map(2024, 7)   # July
    f2  = plot_loading_comparison()
    f3  = plot_monthly_balance()

    log.info("\nDone. Files:")
    for f in [f1a, f1b, f2, f3]:
        if f:
            log.info("  %s", f)
