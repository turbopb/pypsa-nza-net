#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_ni_network_loading.py
Produces individual NI network loading PNGs:
    ni_network_loading_january2024.png
    ni_network_loading_july2024.png
"""

import os, sys, logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_root = os.environ.get("PYPSA_NZA_ROOT", "")
if not _root:
    log.error("PYPSA_NZA_ROOT not set"); sys.exit(1)

workspace  = Path(_root) / "pypsa_nza_workspace"
manual_dir = workspace / "data" / "manual"
verif_dir  = workspace / "outputs" / "verification_pf" / "NI"
fig_dir    = workspace / "outputs" / "figures" / "validation"
fig_dir.mkdir(parents=True, exist_ok=True)

PROJ      = ccrs.PlateCarree()
NI_EXTENT = [172.4, 178.8, -41.8, -34.2]
MONTHS    = ["jan","feb","mar","apr","may","jun",
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
    month_label = "{}_{}".format(MONTHS[month-1], year)
    path = verif_dir / str(year) / month_label / "line_loading.csv"
    if not path.exists():
        log.warning("Not found: %s", path); return None
    return pd.read_csv(str(path), index_col=0)

def get_coords(site, nodes):
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

def plot_network_map(year=2024, month=1):
    nodes   = load_nodes()
    lines   = load_lines()
    loading = load_loading(year, month)
    if loading is None: return

    month_name = ["January","February","March","April","May","June",
                  "July","August","September","October","November",
                  "December"][month-1]

    cmap = matplotlib.colormaps["RdYlGn_r"]
    norm = mcolors.Normalize(vmin=0, vmax=100)

    fig = plt.figure(figsize=(11, 14), dpi=200)
    ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
    ax.set_extent(NI_EXTENT, crs=PROJ)
    ax.add_feature(cfeature.LAND,      facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#d6eaf8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#555555", zorder=1)
    ax.add_feature(cfeature.RIVERS,    linewidth=0.4, edgecolor="#aed6f1",
                   facecolor="none", zorder=1)
    ax.add_feature(cfeature.LAKES,     facecolor="#aed6f1", edgecolor="#aed6f1", zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}

    plotted_buses = set()
    line_loadings = []

    for _, row in lines.iterrows():
        line_id = str(row["line"]).strip()
        b0 = str(row["bus0"]).strip().upper()
        b1 = str(row["bus1"]).strip().upper()
        x0, y0 = get_coords(b0, nodes)
        x1, y1 = get_coords(b1, nodes)
        if None in (x0, y0, x1, y1): continue
        plotted_buses.add(b0); plotted_buses.add(b1)
        pct = float(loading.loc[line_id, "mean_loading_pct"]) if line_id in loading.index else 0.0
        mx  = float(loading.loc[line_id, "max_loading_pct"])  if line_id in loading.index else 0.0
        color = cmap(norm(pct))
        lw    = 1.0 + (pct / 100) ** 1.5 * 8.0
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                solid_capstyle="round", transform=PROJ, zorder=3, alpha=0.9)
        line_loadings.append((pct, line_id, x0, y0, x1, y1, mx))

    # Single annotation -- highest loaded line only
    if line_loadings:
        line_loadings.sort(reverse=True)
        pct, line_id, x0, y0, x1, y1, mx = line_loadings[0]
        mid_lon = (x0 + x1) / 2
        mid_lat = (y0 + y1) / 2
        label = "{:s}  {:.0f}% mean\n(max {:.0f}%)".format(line_id, pct, mx)
        ax.annotate(label, xy=(mid_lon, mid_lat), xytext=(177.5, -38.5),
                    fontsize=10, color="#1a1a1a", fontweight="bold",
                    xycoords=PROJ._as_mpl_transform(ax),
                    textcoords=PROJ._as_mpl_transform(ax),
                    arrowprops=dict(arrowstyle="-", color="#333333", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="#666666", alpha=0.95),
                    zorder=8, ha="left")

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
        "OKE","KIN","TMI","MTM","WKO","WHU","OHW","CBG"
    }
    key_labels = {
        "HAY","BPE","WIL","SFD","NAP","THI","WRK","KAW","ARI","TKU",
        "LTN","MTI","OHK","RPO","TUI","TWH","HLY","HOB","MAT","WAI",
        "TGA","WKM","KOE","HWA","TAB","OKI","WPA","ONG","NPK","TMN",
        "ARA","ATI","ROT","TWC","WVY","WWD","TRK","OKE","KIN","EDG",
        "GLN","MKE","KPA","WDV"
    }

    for bus in plotted_buses:
        x, y = get_coords(bus, nodes)
        if x is None: continue
        base = bus.rsplit("_", 1)[0] if not bus.startswith("JCT_") else bus
        if base in gen_buses:
            ms, mk, mc, zord = 10, "o", "#1565C0", 4
        elif base in load_buses:
            ms, mk, mc, zord = 8,  "s", "#B71C1C", 4
        else:
            ms, mk, mc, zord = 5,  "o", "#546E7A", 3
        ax.plot(x, y, mk, color=mc, markersize=ms, zorder=zord,
                markeredgecolor="white", markeredgewidth=0.8, transform=PROJ)
        if base in key_labels:
            ax.annotate(base, xy=(x, y), xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8, color="#111111", fontweight="bold",
                        transform=PROJ, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.12",
                                  facecolor="white", edgecolor="none", alpha=0.8))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.3, pad=0.02, aspect=20, location="right")
    cbar.set_label("Mean line loading (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks([0, 25, 50, 75, 100])

    legend_elements = [
        mlines.Line2D([0],[0], marker="o", color="w", markerfacecolor="#1565C0",
                      markersize=10, markeredgecolor="white", label="Generation bus"),
        mlines.Line2D([0],[0], marker="s", color="w", markerfacecolor="#B71C1C",
                      markersize=9,  markeredgecolor="white", label="Load / substation bus"),
        mlines.Line2D([0],[0], marker="o", color="w", markerfacecolor="#546E7A",
                      markersize=6,  markeredgecolor="white", label="Junction / transformer bus"),
        mlines.Line2D([0],[0], color=cmap(norm(10)), linewidth=2.0, label="Lightly loaded"),
        mlines.Line2D([0],[0], color=cmap(norm(50)), linewidth=3.5, label="Moderately loaded"),
        mlines.Line2D([0],[0], color=cmap(norm(90)), linewidth=6.0, label="Heavily loaded"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              framealpha=0.95, edgecolor="#bbbbbb", borderpad=0.8)

    ax.set_title(
        "North Island Transmission Network\n"
        "DC Power Flow Validation -- {} {}\n"
        "Lines coloured and weighted by mean loading (% of rated capacity)"
        .format(month_name, year),
        fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    out = fig_dir / "ni_network_loading_{}{}.png".format(month_name.lower(), year)
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s", out)

if __name__ == "__main__":
    plot_network_map(2024, 1)
    plot_network_map(2024, 7)
