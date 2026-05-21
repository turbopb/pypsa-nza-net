#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_network_topology.py

Generate publication-quality PNG maps of the SI transmission network
topology using cartopy for the NZ coastline background.

Produces four figures:
    1. si_buses.png           -- all SI buses coloured by voltage level
    2. si_topology.png        -- buses + transmission lines
    3. si_transformers.png    -- transformer locations highlighted
    4. si_generators.png      -- generator locations coloured by carrier

Outputs written to:
    pypsa_nza_workspace/outputs/figures/network/

Usage
-----
    python plot_network_topology.py [--island SI] [--dpi 300]

Author: Phillippe Bruneau
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =============================================================================
# STYLE CONSTANTS
# =============================================================================

# Voltage level colours
VOLT_COLOURS = {
    220: "#c0392b",   # red
    110: "#2980b9",   # blue
    350: "#8e44ad",   # purple (DC)
}
VOLT_LABELS = {
    220: "220 kV",
    110: "110 kV",
    350: "HVDC",
}

# Carrier colours for generators
CARRIER_COLOURS = {
    "hydro":      "#2980b9",
    "solar_pv":   "#f39c12",
    "wind":       "#27ae60",
    "geothermal": "#e74c3c",
    "gas":        "#7f8c8d",
    "ocgt":       "#95a5a6",
    "ccgt":       "#bdc3c7",
    "coal_gas":   "#2c3e50",
    "diesel":     "#e67e22",
}

# Map extent for South Island [lon_min, lon_max, lat_min, lat_max]
SI_EXTENT = [166.0, 174.5, -46.8, -40.3]
# Map extent for North Island
NI_EXTENT = [172.5, 178.7, -41.7, -34.3]

PROJ = ccrs.PlateCarree()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(workspace: Path, island: str = "SI") -> dict:
    """Load all static network data needed for plotting."""
    manual_dir   = workspace / "data" / "manual"
    static_dir   = workspace / "data" / "processed" / "static"

    # Nodes
    df_nodes = pd.read_csv(manual_dir / "nodes.csv", dtype=str)
    for col in df_nodes.columns:
        if df_nodes[col].dtype == object:
            df_nodes[col] = df_nodes[col].str.strip()
    df_nodes["volts"] = pd.to_numeric(df_nodes["volts"], errors="coerce")
    df_nodes["long"]  = pd.to_numeric(df_nodes["long"],  errors="coerce")
    df_nodes["lat"]   = pd.to_numeric(df_nodes["lat"],   errors="coerce")

    # Filter to island -- keep ALL nodes including junction points
    # Junction nodes are needed for coordinate lookup when drawing lines
    # that pass through them. They are excluded from bus markers but
    # must be in the coordinate lookup table.
    ELECTRODE = {"BGR", "THW"}
    df_nodes = df_nodes[
        (df_nodes["island"] == island) &
        (~df_nodes["site"].isin(ELECTRODE))
    ].copy()
    log.info("Nodes loaded: %d %s sites (including junction nodes)",
             len(df_nodes), island)

    # Lines
    lines_file = "lines_data_ni.csv" if island == "NI" else "lines_data_si.csv"
    df_lines = pd.read_csv(manual_dir / lines_file, dtype=str)
    for col in df_lines.columns:
        if df_lines[col].dtype == object:
            df_lines[col] = df_lines[col].str.strip()
    df_lines["volts"]    = pd.to_numeric(df_lines["volts"],    errors="coerce")
    df_lines["distance"] = pd.to_numeric(df_lines["distance"], errors="coerce")
    df_lines = df_lines[
        df_lines["volts"].notna() &
        (df_lines["volts"] >= 110) &
        (df_lines["volts"] < 350)
    ].copy()
    log.info("Lines loaded: %d AC lines", len(df_lines))

    # Generator registry
    df_gen = pd.read_csv(static_dir / "generator_registry.csv", dtype=str)
    for col in df_gen.columns:
        if df_gen[col].dtype == object:
            df_gen[col] = df_gen[col].str.strip()
    df_gen["p_nom_mw"] = pd.to_numeric(df_gen["p_nom_mw"], errors="coerce")

    # Node lookup
    # node_lkp includes ALL nodes (for coordinate lookup when drawing lines)
    node_lkp = df_nodes.set_index("site")[["long", "lat", "volts", "name"]]

    # df_nodes_display excludes junction nodes from bus markers on maps
    df_nodes = df_nodes[~df_nodes["site"].str.startswith("JCT_")].copy()

    # Multi-voltage sites from lines
    site_volts = {}
    for _, row in df_lines.iterrows():
        v = int(row["volts"])
        for col in ("bus0", "bus1"):
            s = str(row[col]).strip().upper()
            site_volts.setdefault(s, set()).add(v)
    multi_volt = {s for s, vs in site_volts.items() if len(vs) > 1}

    return {
        "nodes":     df_nodes,
        "lines":     df_lines,
        "gen":       df_gen,
        "node_lkp":  node_lkp,
        "multi_volt": multi_volt,
    }


# =============================================================================
# MAP SETUP
# =============================================================================

def make_figure(title: str, dpi: int = 300, island: str = "SI") -> tuple:
    """Create a figure with NZ transmission network cartopy axes."""
    fig = plt.figure(figsize=(10, 14), dpi=dpi)
    ax  = fig.add_subplot(1, 1, 1, projection=PROJ)

    ax.set_extent(NI_EXTENT if island == "NI" else SI_EXTENT, crs=PROJ)

    # Coastline and land
    ax.add_feature(cfeature.LAND,       facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.OCEAN,      facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.6, edgecolor="#555555", zorder=1)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.4, edgecolor="#aaaaaa", zorder=1)
    ax.add_feature(cfeature.RIVERS,     linewidth=0.3, edgecolor="#aed6f1",
                   facecolor="none", zorder=1)
    ax.add_feature(cfeature.LAKES,      facecolor="#aed6f1", edgecolor="#aed6f1",
                   zorder=1)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels    = False
    gl.right_labels  = False
    gl.xformatter    = LONGITUDE_FORMATTER
    gl.yformatter    = LATITUDE_FORMATTER
    gl.xlabel_style  = {"size": 8}
    gl.ylabel_style  = {"size": 8}

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    return fig, ax


def get_bus_coords(site: str, volts: int,
                   node_lkp: pd.DataFrame,
                   multi_volt: set) -> tuple:
    """Return (long, lat) for a bus, handling multi-voltage suffix."""
    site_upper = site.strip().upper()
    if site_upper in node_lkp.index:
        row = node_lkp.loc[site_upper]
        return row["long"], row["lat"]
    # Try stripping voltage suffix (e.g. ROX_220 -> ROX)
    base = site_upper.rsplit("_", 1)[0]
    if base in node_lkp.index:
        row = node_lkp.loc[base]
        return row["long"], row["lat"]
    return None, None


# =============================================================================
# FIGURE 1: BUSES
# =============================================================================

def plot_buses(data: dict, out_path: Path, dpi: int = 300, island: str = "SI"):
    """Plot all SI buses coloured by voltage level."""
    fig, ax = make_figure(f"{island} Transmission Buses", dpi, island=island)
    df_nodes = data["nodes"]

    for volt, colour in VOLT_COLOURS.items():
        if volt == 350:
            continue
        subset = df_nodes[df_nodes["volts"] == volt]
        if subset.empty:
            continue
        ax.scatter(
            subset["long"], subset["lat"],
            s=60, c=colour, zorder=5,
            edgecolors="white", linewidths=0.5,
            transform=PROJ, label=VOLT_LABELS[volt]
        )

    # Label all buses -- exclude junction nodes
    for site, row in data["node_lkp"].iterrows():
        if pd.isna(row["long"]) or pd.isna(row["lat"]):
            continue
        if site.startswith("JCT_"):
            continue
        ax.annotate(
            site,
            xy=(row["long"], row["lat"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=5.5, color="#333333",
            transform=PROJ,
            zorder=6
        )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
              title="Voltage", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path.name)


# =============================================================================
# FIGURE 2: TOPOLOGY
# =============================================================================

def plot_topology(data: dict, out_path: Path, dpi: int = 300, island: str = "SI"):
    """Plot buses and transmission lines."""
    fig, ax = make_figure(f"{island} Transmission Network Topology", dpi, island=island)
    df_nodes = data["nodes"]
    df_lines = data["lines"]
    node_lkp = data["node_lkp"]
    multi_volt = data["multi_volt"]

    # Draw lines
    for _, row in df_lines.iterrows():
        v    = int(row["volts"])
        col  = VOLT_COLOURS.get(v, "#888888")
        cfg  = str(row.get("config", row.get("cofing", "single"))).strip().lower()
        lw   = 1.8 if cfg == "double" else 1.1

        x0, y0 = get_bus_coords(row["bus0"], v, node_lkp, multi_volt)
        x1, y1 = get_bus_coords(row["bus1"], v, node_lkp, multi_volt)
        if None in (x0, y0, x1, y1):
            log.debug("  Skipping line %s -- coords missing", row["line"])
            continue

        ax.plot([x0, x1], [y0, y1],
                color=col, linewidth=lw * 1.8, alpha=0.85,
                transform=PROJ, zorder=3)

    # Draw buses on top
    for volt, colour in VOLT_COLOURS.items():
        if volt == 350:
            continue
        subset = df_nodes[df_nodes["volts"] == volt]
        ax.scatter(
            subset["long"], subset["lat"],
            s=45, c=colour, zorder=5,
            edgecolors="white", linewidths=0.5,
            transform=PROJ
        )

    # Add bus labels to topology -- exclude junction nodes
    node_lkp = data["node_lkp"]
    line_buses = set(data["lines"]["bus0"].str.upper().tolist() +
                     data["lines"]["bus1"].str.upper().tolist())
    for site, row in node_lkp.iterrows():
        if pd.isna(row["long"]) or pd.isna(row["lat"]):
            continue
        if site.startswith("JCT_"):
            continue
        if site not in line_buses:
            continue
        ax.annotate(
            site,
            xy=(row["long"], row["lat"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=5.5, color="#333333",
            transform=PROJ,
            zorder=6
        )

    # Legend
    handles = []
    for volt, col in VOLT_COLOURS.items():
        if volt == 350:
            continue
        handles.append(mlines.Line2D([], [], color=col, linewidth=2.0,
                                     label=VOLT_LABELS[volt]))
    handles.append(mlines.Line2D([], [], color=VOLT_COLOURS[220],
                                 linewidth=3.0, label="Double circuit"))
    handles.append(mlines.Line2D([], [], color=VOLT_COLOURS[110],
                                 linewidth=1.8, label="Single circuit"))
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.9, title="Line type", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path.name)


# =============================================================================
# FIGURE 3: TRANSFORMERS
# =============================================================================

def plot_transformers(data: dict, out_path: Path, dpi: int = 300, island: str = "SI"):
    """Plot network topology with transformer locations highlighted."""
    fig, ax = make_figure(
        f"{island} Transmission Network -- Transformer Locations", dpi, island=island)
    df_lines   = data["lines"]
    node_lkp   = data["node_lkp"]
    multi_volt = data["multi_volt"]
    df_nodes   = data["nodes"]

    # Draw lines (greyed out)
    for _, row in df_lines.iterrows():
        v   = int(row["volts"])
        x0, y0 = get_bus_coords(row["bus0"], v, node_lkp, multi_volt)
        x1, y1 = get_bus_coords(row["bus1"], v, node_lkp, multi_volt)
        if None in (x0, y0, x1, y1):
            continue
        ax.plot([x0, x1], [y0, y1],
                color="#cccccc", linewidth=1.2, alpha=0.7,
                transform=PROJ, zorder=2)

    # Regular buses
    for volt, colour in VOLT_COLOURS.items():
        if volt == 350:
            continue
        subset = df_nodes[df_nodes["volts"] == volt]
        ax.scatter(subset["long"], subset["lat"],
                   s=30, c=colour, alpha=0.6, zorder=3,
                   edgecolors="white", linewidths=0.4,
                   transform=PROJ)

    # Transformer sites (multi-voltage)
    for site in sorted(multi_volt):
        if site not in node_lkp.index:
            continue
        row = node_lkp.loc[site]
        ax.scatter(
            row["long"], row["lat"],
            s=80, c="#e74c3c", marker="^",
            edgecolors="#7b241c", linewidths=0.8,
            zorder=6, transform=PROJ
        )
        ax.annotate(
            site,
            xy=(row["long"], row["lat"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=6.5, color="#7b241c", fontweight="bold",
            transform=PROJ, zorder=7
        )

    # Legend
    handles = [
        mpatches.Patch(facecolor=VOLT_COLOURS[220], label="220 kV bus"),
        mpatches.Patch(facecolor=VOLT_COLOURS[110], label="110 kV bus"),
        mlines.Line2D([], [], marker="^", color="w",
                      markerfacecolor="#e74c3c", markeredgecolor="#7b241c",
                      markersize=9, label="220/110 kV transformer"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path.name)


# =============================================================================
# FIGURE 4: GENERATORS
# =============================================================================

def plot_generators(data: dict, out_path: Path, dpi: int = 300, island: str = "SI"):
    """Plot network topology with generator locations coloured by carrier."""
    fig, ax = make_figure(
        f"{island} Transmission Network -- Generator Locations", dpi, island=island)
    df_lines   = data["lines"]
    node_lkp   = data["node_lkp"]
    multi_volt = data["multi_volt"]
    df_nodes   = data["nodes"]
    df_gen     = data["gen"]

    # Draw lines
    for _, row in df_lines.iterrows():
        v   = int(row["volts"])
        x0, y0 = get_bus_coords(row["bus0"], v, node_lkp, multi_volt)
        x1, y1 = get_bus_coords(row["bus1"], v, node_lkp, multi_volt)
        if None in (x0, y0, x1, y1):
            continue
        ax.plot([x0, x1], [y0, y1],
                color="#cccccc", linewidth=1.2, alpha=0.7,
                transform=PROJ, zorder=2)

    # Regular buses
    ax.scatter(df_nodes["long"], df_nodes["lat"],
               s=25, c="#aaaaaa", alpha=0.4, zorder=3,
               edgecolors="white", linewidths=0.3,
               transform=PROJ)

    # Generators -- join to nodes for coordinates
    si_sites = set(df_nodes["site"].str.upper())
    gen_handles = {}

    for _, row in df_gen.iterrows():
        site    = row["site"].strip().upper()
        carrier = row["carrier"].strip()
        p_nom   = row["p_nom_mw"]

        # Only SI sites
        if site not in si_sites:
            continue

        # Get coordinates -- handle multi-voltage suffix
        base = site.rsplit("_", 1)[0]
        coord_site = site if site in node_lkp.index else base
        if coord_site not in node_lkp.index:
            log.debug("  Generator %s: no coords", site)
            continue

        coords = node_lkp.loc[coord_site]
        colour = CARRIER_COLOURS.get(carrier, "#888888")
        size   = max(30, float(p_nom) * 0.15) if pd.notna(p_nom) else 30

        ax.scatter(
            coords["long"], coords["lat"],
            s=size, c=colour,
            edgecolors="white", linewidths=0.5,
            zorder=6, transform=PROJ
        )

        if carrier not in gen_handles:
            gen_handles[carrier] = mpatches.Patch(
                facecolor=colour,
                label=carrier.replace("_", " ").title()
            )

    # Legend
    handles = list(gen_handles.values())
    handles.append(mlines.Line2D([], [], marker="o", color="w",
                                 markerfacecolor="#555555", markersize=6,
                                 label="Size ~ capacity"))
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.9, title="Carrier", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path.name)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot SI transmission network topology maps"
    )
    parser.add_argument("--island", default="SI",
                        help="Island to plot: SI or NI (default: SI)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output resolution in DPI (default: 300)")
    args = parser.parse_args()

    _root = os.environ.get("PYPSA_NZA_ROOT", "")
    if not _root:
        log.error("PYPSA_NZA_ROOT not set. See SETUP.md.")
        sys.exit(1)

    workspace = Path(_root) / "pypsa_nza_workspace"
    out_dir   = workspace / "outputs" / "figures" / "network"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("PLOT NETWORK TOPOLOGY")
    log.info("=" * 60)
    log.info("Island    : %s", args.island)
    log.info("DPI       : %d", args.dpi)
    log.info("Output dir: %s", out_dir)

    data = load_data(workspace, args.island)

    plot_buses(
        data,
        out_dir / f"{args.island.lower()}_buses.png",
        args.dpi,
        island=args.island,
    )
    plot_topology(
        data,
        out_dir / f"{args.island.lower()}_topology.png",
        args.dpi,
        island=args.island,
    )
    plot_transformers(
        data,
        out_dir / f"{args.island.lower()}_transformers.png",
        args.dpi,
        island=args.island,
    )
    plot_generators(
        data,
        out_dir / f"{args.island.lower()}_generators.png",
        args.dpi,
        island=args.island,
    )

    log.info("\nDone. Four figures written to: %s", out_dir)


if __name__ == "__main__":
    main()
