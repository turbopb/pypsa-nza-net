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
    SI network map with lines coloured by mean loading.
    Uses cartopy geographic background matching NI style.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        import matplotlib.lines as mlines
    except ImportError:
        log.error("cartopy not available -- cannot generate network map")
        return

    log.info("Figure 1: SI network map (month=%d)", month)

    nodes    = load_nodes()
    lines_df = pd.read_csv(str(manual_dir / "lines_data_si.csv"), dtype=str)
    lines_df.columns = lines_df.columns.str.strip()
    loading  = load_loading(year, month)
    if loading is None:
        log.error("No loading data for %d/%d", year, month)
        return

    PROJ      = ccrs.PlateCarree()
    SI_EXTENT = [166.0, 174.5, -46.8, -40.3]

    def get_coords(site):
        site = str(site).strip().upper()
        if site in nodes.index:
            row = nodes.loc[site]
            if pd.notna(row["lat"]) and pd.notna(row["long"]):
                return float(row["long"]), float(row["lat"])
        base = site.rsplit("_", 1)[0] if not site.startswith("JCT_") else site
        if base in nodes.index:
            row = nodes.loc[base]
            if pd.notna(row["lat"]) and pd.notna(row["long"]):
                return float(row["long"]), float(row["lat"])
        return None, None

    cmap = matplotlib.colormaps["RdYlGn_r"]
    norm = mcolors.Normalize(vmin=0, vmax=100)

    fig = plt.figure(figsize=(14, 10), dpi=300)
    ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
    ax.set_extent(SI_EXTENT, crs=PROJ)
    ax.add_feature(cfeature.LAND,      facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#555555", zorder=1)
    ax.add_feature(cfeature.RIVERS,    linewidth=0.3, edgecolor="#aed6f1",
                   facecolor="none", zorder=1)
    ax.add_feature(cfeature.LAKES,     facecolor="#aed6f1", edgecolor="#aed6f1",
                   zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    plotted_buses = set()
    line_loadings = []

    # --- Draw lines ---
    for _, row in lines_df.iterrows():
        line_id = str(row.iloc[0]).strip()
        b0 = str(row.iloc[3]).strip().upper()
        b1 = str(row.iloc[4]).strip().upper()
        x0, y0 = get_coords(b0)
        x1, y1 = get_coords(b1)
        if None in (x0, y0, x1, y1):
            continue
        plotted_buses.add(b0)
        plotted_buses.add(b1)
        pct = float(loading.loc[line_id, "mean_loading_pct"]) \
              if line_id in loading.index else 0.0
        mx  = float(loading.loc[line_id, "max_loading_pct"]) \
              if line_id in loading.index else 0.0
        color = cmap(norm(pct))
        lw    = 1.0 + (pct / 100) ** 1.5 * 7.0
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                solid_capstyle="round", transform=PROJ, zorder=3, alpha=0.9)
        line_loadings.append((pct, line_id, x0, y0, x1, y1, mx))

    # --- Annotate top 3 lines -- boxes in Tasman Sea west of network ---
    line_loadings.sort(reverse=True)
    sea_positions = [
        (166.4, -41.8),
        (166.4, -43.0),
        (166.4, -44.2),
    ]
    for i, (pct, line_id, x0, y0, x1, y1, mx) in enumerate(line_loadings[:3]):
        mid_lon = (x0 + x1) / 2
        mid_lat = (y0 + y1) / 2
        txt_lon, txt_lat = sea_positions[i]
        label = "{:s}  {:.0f}% mean\n(max {:.0f}%)".format(line_id, pct, mx)
        ax.annotate(
            label,
            xy=(mid_lon, mid_lat),
            xytext=(txt_lon, txt_lat),
            fontsize=8.5, color="#1a1a1a", fontweight="bold",
            xycoords=PROJ._as_mpl_transform(ax),
            textcoords=PROJ._as_mpl_transform(ax),
            arrowprops=dict(arrowstyle="-", color="#333333", lw=0.9),
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#666666", alpha=0.95, linewidth=0.8),
            zorder=8, ha="left"
        )

    # --- Draw buses ---
    gen_buses  = {"MAN","BEN","CYD","ROX","OHA","OHB","OHC","AVI",
                  "TKB","ARG","BWK","GOR","WTK","COL","TKA","NMA","STK"}
    load_buses = {"ISL","TWI","INV","TIM","BLN","HWB","NWD","TWZ",
                  "CML","KIK","ABY","ORO"}
    key_labels = {"MAN","BEN","CYD","ROX","OHA","OHB","OHC","AVI","TKB",
                  "ISL","TWI","INV","TIM","BLN","HWB","NWD","KIK","STK",
                  "NMA","WTK","CML","TWZ","ARG","GOR","TKA","ABY","ORO",
                  "MCH","NSY"}

    for bus in plotted_buses:
        x, y = get_coords(bus)
        if x is None:
            continue
        base = bus.rsplit("_", 1)[0] if not bus.startswith("JCT_") else bus
        if base in gen_buses:
            ms, mk, mc, zord = 8, "o", "#1565C0", 4
        elif base in load_buses:
            ms, mk, mc, zord = 7, "s", "#B71C1C", 4
        else:
            ms, mk, mc, zord = 4, "o", "#546E7A", 3
        ax.plot(x, y, mk, color=mc, markersize=ms, zorder=zord,
                markeredgecolor="white", markeredgewidth=0.8, transform=PROJ)
        if base in key_labels:
            ax.annotate(base, xy=(x, y), xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=6.5, color="#111111", fontweight="bold",
                        transform=PROJ, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.8))

    # --- Colourbar ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.25, pad=0.02,
                        aspect=16, location="right")
    cbar.set_label("Mean line loading (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0, 25, 50, 75, 100])

    # --- Legend ---
    legend_elements = [
        mlines.Line2D([0],[0], marker="o", color="w",
                      markerfacecolor="#1565C0", markersize=9,
                      markeredgecolor="white", label="Generation bus"),
        mlines.Line2D([0],[0], marker="s", color="w",
                      markerfacecolor="#B71C1C", markersize=8,
                      markeredgecolor="white", label="Load / substation bus"),
        mlines.Line2D([0],[0], marker="o", color="w",
                      markerfacecolor="#546E7A", markersize=6,
                      markeredgecolor="white", label="Junction / transformer bus"),
        mlines.Line2D([0],[0], color=cmap(norm(10)),
                      linewidth=2.0, label="Lightly loaded line"),
        mlines.Line2D([0],[0], color=cmap(norm(50)),
                      linewidth=3.5, label="Moderately loaded line"),
        mlines.Line2D([0],[0], color=cmap(norm(90)),
                      linewidth=5.0, label="Heavily loaded line"),
    ]
    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=8, framealpha=0.95,
              edgecolor="#bbbbbb", borderpad=0.8)

    month_name = ["January","February","March","April","May","June",
                  "July","August","September","October","November",
                  "December"][month-1]
    ax.set_title(
        "South Island Transmission Network\n"
        "DC Power Flow Validation -- {} {}\n"
        "Lines coloured and weighted by mean loading (% of rated capacity)"
        .format(month_name, year),
        fontsize=11, fontweight="bold", pad=12
    )

    fig.tight_layout()
    out = fig_dir / "si_network_loading_{}{}.png".format(month_name.lower(), year)
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
