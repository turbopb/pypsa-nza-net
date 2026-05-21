#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_ni_network_loading.py

NI network loading maps for thesis appendix.
Matches the style of plot_dc_validation.py SI maps exactly.

Generates:
    ni_network_loading_january2024.png
    ni_network_loading_july2024.png

Usage
-----
    python plot_ni_network_loading.py

Output written to:
    {workspace}/outputs/figures/validation/
"""

import os
import sys
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_root = os.environ.get("PYPSA_NZA_ROOT", "")
if not _root:
    log.error("PYPSA_NZA_ROOT not set")
    sys.exit(1)

workspace  = Path(_root) / "pypsa_nza_workspace"
manual_dir = workspace / "data" / "manual"
verif_dir  = workspace / "outputs" / "verification_pf" / "NI"
fig_dir    = workspace / "outputs" / "figures" / "validation"
fig_dir.mkdir(parents=True, exist_ok=True)

MONTHS = ["jan","feb","mar","apr","may","jun",
          "jul","aug","sep","oct","nov","dec"]

def load_nodes():
    df = pd.read_csv(str(manual_dir / "nodes.csv"), dtype=str)
    df["site"] = df["site"].str.strip().str.upper()
    df["lat"]  = pd.to_numeric(df["lat"],  errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    return df.set_index("site")

def load_lines():
    df = pd.read_csv(str(manual_dir / "lines_data_ni.csv"), dtype=str)
    df.columns = df.columns.str.strip()
    return df

def load_loading(year, month):
    month_label = f"{MONTHS[month-1]}_{year}"
    path = verif_dir / str(year) / month_label / "line_loading.csv"
    if not path.exists():
        log.warning("Not found: %s", path)
        return None
    return pd.read_csv(str(path), index_col=0)


def plot_network_map(year=2024, month=1):
    log.info("NI network map (month=%d)", month)

    nodes    = load_nodes()
    lines_df = load_lines()
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
        line_id = str(row["line"]).strip()
        b0 = str(row["bus0"]).strip().upper()
        b1 = str(row["bus1"]).strip().upper()

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
        lw    = 0.8 + (pct / 100) ** 1.5 * 8.0

        ax.plot([c0[1], c1[1]], [c0[0], c1[0]],
                color=color, linewidth=lw,
                solid_capstyle="round", zorder=2,
                alpha=0.9)

        # Annotate lines with mean loading > 20%
        if pct > 20:
            mid_lat = (c0[0] + c1[0]) / 2
            mid_lon = (c0[1] + c1[1]) / 2
            ax.annotate(
                f"{line_id}  {pct:.0f}% mean\n(max {mx:.0f}%)",
                xy=(mid_lon, mid_lat),
                xytext=(mid_lon + 0.10, mid_lat + 0.03),
                fontsize=6.0, color="#1a1a1a",
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
    gen_buses = {
        "ARA","ARI","ATI","HRP","HWA","JRD","KAW","KOE","KPA","KPO",
        "LTN","MKE","MTI","NAP","OHK","OKI","PPI","ROT","RPO","SFD",
        "TAB","THI","TKU","TUI","TWC","WDV","WHI","WKM","WPA","WRK",
        "WVY","WWD","GLN","EDG"
    }
    load_buses = {
        "HAY","BPE","WIL","HLY","HOB","MAT","WAI","TGA","TMN","NPK",
        "ONG","WGN","BRK","MTN","OKN","TNS","MTR","OTA","PAK","SVL",
        "WEL","ALB","HPI","MNG","BOB","WRD","TAK","GYT","MHO","TRK",
        "OKE","EDG","KIN","TMI","MTM","WKO","WHU","OHW","CBG","KPO",
        "RPO","SFD","NAP","KAW"
    }

    # Key buses to label
    key_labels = {
        "HAY","BPE","WIL","SFD","NAP","THI","WRK","KAW","ARI","TKU",
        "LTN","MTI","OHK","RPO","TUI","TWH","HLY","HOB","MAT","WAI",
        "TGA","WKM","GLN","KOE","HWA","MKE","TAB","EDG","OKI","WPA",
        "ONG","NPK","TMN","ARA","ATI","KPA","ROT","TWC","WDV","WVY",
        "WWD","WHI","TRK","OKE","KIN"
    }

    for bus in plotted_buses:
        c = get_coords(bus)
        if c is None:
            continue
        base = bus.rsplit("_", 1)[0] if not bus.startswith("JCT_") else bus

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
            ax.text(c[1] + 0.06, c[0] + 0.02,
                    base,
                    fontsize=7.0,
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
        f"North Island Transmission Network\n"
        f"DC Power Flow Validation -- {month_name} {year}\n"
        f"Lines coloured and weighted by mean loading (% of rated capacity)",
        fontsize=10, pad=14, fontweight="bold"
    )
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°S)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.5, color="#888888")
    ax.set_xlim(172.4, 178.8)
    ax.set_ylim(-41.8, -34.2)

    fig.tight_layout()
    out = fig_dir / f"ni_network_loading_{month_name.lower()}{year}.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


if __name__ == "__main__":
    plot_network_map(2024, 1)
    plot_network_map(2024, 7)
