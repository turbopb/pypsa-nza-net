#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dc_power_flow.py

Standalone numpy DC power flow for the SI transmission network.

Solves B * theta = P directly using numpy linear algebra.
No PyPSA API involvement beyond loading the network topology
and electrical parameters.

Theory
------
The DC power flow is a linearisation of the full AC power flow:
- Voltage magnitudes assumed = 1.0 pu everywhere
- Resistance neglected (lossless)
- Small angle approximation: sin(theta) ~ theta

The susceptance matrix B is built from line reactances:
    B[i,i] += 1/x_ij  (diagonal)
    B[i,j] -= 1/x_ij  (off-diagonal)

Net injection at each bus:
    P[i] = generation[i] - load[i]  (MW)

System of equations:
    B * theta = P

Slack bus (BEN) has theta fixed at 0 -- row/col removed before solving.

Line flows:
    p_line = (theta_i - theta_j) / x_ij  (per unit, then scaled to MW)

Usage
-----
    python dc_power_flow.py [--year 2024] [--month 1]

Outputs
-------
    {workspace}/outputs/verification/{year}/{month}/
        dc_line_loading.csv      -- per-line max/mean/p95 loading
        dc_power_flow.png        -- line loading bar chart
        dc_bus_injections.csv    -- net injection at each bus (mean)
        dc_line_flows.csv        -- full timeseries of line flows

Author: pypsa-nza-net
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

import pypsa

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MONTHS = ["jan","feb","mar","apr","may","jun",
          "jul","aug","sep","oct","nov","dec"]

CUSTOM_TRANSFORMER_TYPES = {
    "630MVA 220/110kV": {
        "f_nom":        50.0,
        "vn_hv":       220.0,
        "vn_lv":       110.0,
        "vsc":          12.0,
        "vscr":          0.4,
        "pfe":           0.0,
        "i0":            0.0,
        "s_nom":       630.0,
        "tap_side":      0,
        "tap_neutral":   0,
        "tap_min":      -9,
        "tap_max":       9,
        "tap_step":      1.5,
    }
}


# =============================================================================
# NETWORK LOADING
# =============================================================================

def load_network(network_dir: Path) -> pypsa.Network:
    n = pypsa.Network()
    n.import_from_csv_folder(str(network_dir))
    _ref = pypsa.Network()
    n.transformer_types = _ref.transformer_types
    n.line_types        = _ref.line_types
    for name, params in CUSTOM_TRANSFORMER_TYPES.items():
        if name not in n.transformer_types.index:
            n.transformer_types.loc[name] = params
    n.calculate_dependent_values()

    # Recompute s_nom from line types -- not persisted to CSV
    import math as _math
    for line in n.lines.index:
        if n.lines.loc[line, "s_nom"] > 0:
            continue
        lt    = n.lines.loc[line, "type"]
        v_nom = float(n.lines.loc[line, "v_nom"]) \
                if "v_nom" in n.lines.columns else 220.0
        n_par = float(n.lines.loc[line, "num_parallel"])
        if lt and lt in n.line_types.index:
            i_nom = float(n.line_types.loc[lt, "i_nom"])
            n.lines.loc[line, "s_nom"] = \
                _math.sqrt(3) * v_nom * i_nom * n_par
        else:
            n.lines.loc[line, "s_nom"] = 200.0 if v_nom >= 200 else 100.0

    log.info("  s_nom range: %.1f -- %.1f MW",
             n.lines["s_nom"].min(), n.lines["s_nom"].max())

    # Load loads_t.p_set timeseries directly from CSV if present
    pset_path = network_dir / "loads-p_set.csv"
    if pset_path.exists() and n.loads_t.p_set.empty:
        df_pset = pd.read_csv(str(pset_path), index_col=0)
        # Index is integer positions -- assign snapshot index directly
        if len(df_pset) == len(n.snapshots):
            df_pset.index = n.snapshots
        n.loads_t.p_set = df_pset
        log.info("  Loaded loads_t.p_set: %d loads x %d timesteps  mean=%.1f MW",
                 len(df_pset.columns), len(df_pset),
                 df_pset.sum(axis=1).mean())

    log.info("  Buses       : %d", len(n.buses))
    log.info("  Lines       : %d", len(n.lines))
    log.info("  Transformers: %d", len(n.transformers))
    log.info("  Generators  : %d", len(n.generators))
    log.info("  Loads       : %d", len(n.loads))
    log.info("  Mean total load: %.1f MW",
             n.loads_t.p_set.sum(axis=1).mean()
             if not n.loads_t.p_set.empty else 0.0)
    return n


# =============================================================================
# TIMESERIES LOADING
# =============================================================================

def load_timeseries(workspace: Path,
                    year: int,
                    month: int,
                    snapshots: pd.DatetimeIndex):
    """Load GIP, GXP, and HVDC timeseries sliced to snapshots."""
    annual_dir = workspace / "data" / "processed" / "annual"
    hvdc_dir   = (workspace / "data" / "processed"
                  / str(year) / "hvdc" / "dc_MW")

    # Load SI site list for island filtering
    nodes_path = workspace / "data" / "manual" / "nodes.csv"
    si_sites = set()
    if nodes_path.exists():
        df_n = pd.read_csv(str(nodes_path), dtype=str)
        for _, row in df_n.iterrows():
            if str(row.get("island", "")).strip().upper() == "SI":
                si_sites.add(str(row.get("site", "")).strip().upper())

    # Generation (GIP import timeseries) -- filter to SI sites only
    imp_path  = annual_dir / f"{year}_import_sites_all.csv"
    df_import = pd.read_csv(imp_path, index_col=0, parse_dates=True)
    df_import.index = df_import.index.tz_localize(None)
    si_imp_cols = [c for c in df_import.columns
                   if c.strip().upper() in si_sites]
    df_import = df_import[si_imp_cols].reindex(snapshots).fillna(0.0)
    log.info("  SI GIP sites: %d  mean=%.1f MW",
             len(df_import.columns),
             df_import.sum(axis=1).mean())

    # HVDC
    hvdc_path = hvdc_dir / f"{year}{month:02d}_dc_MW.csv"
    if hvdc_path.exists():
        df_hvdc = pd.read_csv(hvdc_path, index_col=0, parse_dates=True)
        df_hvdc.index = df_hvdc.index.tz_localize(None)
        hvdc = df_hvdc["BEN2201"].reindex(snapshots).fillna(0.0) \
               if "BEN2201" in df_hvdc.columns \
               else pd.Series(0.0, index=snapshots)
    else:
        log.warning("  HVDC file not found -- set to zero")
        hvdc = pd.Series(0.0, index=snapshots)
    log.info("  HVDC mean: %.1f MW", hvdc.mean())

    return df_import, hvdc


# =============================================================================
# BUILD SUSCEPTANCE MATRIX
# =============================================================================

def build_B_matrix(n: pypsa.Network,
                   ac_buses: list,
                   bus_idx: dict) -> np.ndarray:
    """
    Build the bus susceptance matrix B from line and transformer reactances.
    Uses x_pu (per-unit reactance on 100 MVA base) from calculate_dependent_values().
    """
    n_buses = len(ac_buses)
    B = np.zeros((n_buses, n_buses))

    # AC transmission lines
    skipped_lines = 0
    for line in n.lines.index:
        b0   = n.lines.loc[line, "bus0"]
        b1   = n.lines.loc[line, "bus1"]
        x_pu = float(n.lines.loc[line, "x_pu_eff"]
                     if "x_pu_eff" in n.lines.columns
                     else n.lines.loc[line, "x_pu"])
        if x_pu <= 0 or b0 not in bus_idx or b1 not in bus_idx:
            skipped_lines += 1
            continue
        b = 1.0 / x_pu
        i, j = bus_idx[b0], bus_idx[b1]
        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b

    # Transformers
    skipped_trafos = 0
    for trafo in n.transformers.index:
        b0   = n.transformers.loc[trafo, "bus0"]
        b1   = n.transformers.loc[trafo, "bus1"]
        x_pu = float(n.transformers.loc[trafo, "x_pu"]
                     if "x_pu" in n.transformers.columns
                     else 0.0)
        if x_pu <= 0 or b0 not in bus_idx or b1 not in bus_idx:
            skipped_trafos += 1
            continue
        b = 1.0 / x_pu
        i, j = bus_idx[b0], bus_idx[b1]
        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b

    log.info("  B matrix built: %dx%d", n_buses, n_buses)
    log.info("  Lines included: %d  skipped: %d",
             len(n.lines) - skipped_lines, skipped_lines)
    log.info("  Trafos included: %d  skipped: %d",
             len(n.transformers) - skipped_trafos, skipped_trafos)
    return B


# =============================================================================
# BUILD NET INJECTION VECTOR
# =============================================================================

def build_injection_matrix(n: pypsa.Network,
                            df_import: pd.DataFrame,
                            hvdc: pd.Series,
                            ac_buses: list,
                            bus_idx: dict,
                            network_dir: Path,
                            node_coords: dict) -> np.ndarray:
    """
    Build net injection matrix P (snapshots x buses) in MW.

    P[t, i] = generation[t, i] + hvdc[t, i] - load[t, i]

    Generation: from measured GIP timeseries (transmission sites only).
                Multi-voltage sites split equally between buses.
    HVDC:       added as positive injection at BEN bus.
    Load:       from network loads_t.p_set (all GXPs aggregated).
    """
    n_snap  = len(n.snapshots)
    n_buses = len(ac_buses)
    P       = np.zeros((n_snap, n_buses))

    # Group generators by site for multi-voltage equal split
    site_gens = {}
    for gen in n.generators.index:
        bus  = n.generators.loc[gen, "bus"]
        site = bus.rsplit("_", 1)[0] if "_" in bus else bus
        site_gens.setdefault(site, [])
        site_gens[site].append((gen, bus))

    gen_total = np.zeros(n_snap)
    for site, gen_bus_list in site_gens.items():
        if site not in df_import.columns:
            continue
        fraction = 1.0 / len(gen_bus_list)
        for gen, bus in gen_bus_list:
            if bus not in bus_idx:
                continue
            idx    = bus_idx[bus]
            p_vals = df_import[site].values * fraction
            P[:, idx] += p_vals
            gen_total  += p_vals

    log.info("  Mean transmission generation: %.1f MW", gen_total.mean())

    # Remaining SI GIP sites not already handled as model generators.
    # These are either transmission buses without explicit generators,
    # or sub-transmission sites. All df_import columns are SI (filtered
    # in load_timeseries), so no island check needed here.
    import math as _math

    # AC bus name lookup (strip voltage suffix)
    bus_base = {}
    for bus in ac_buses:
        base = bus.rsplit("_", 1)[0] if "_" in bus else bus
        bus_base[base.upper()] = bus
        bus_base[bus.upper()]  = bus

    # Sites already handled as model generators
    gen_sites = set(site_gens.keys())

    sub_total = np.zeros(n_snap)
    sub_count = 0

    for site in df_import.columns:
        site_up = site.strip().upper()
        if site_up in gen_sites:
            continue  # already handled

        p_vals = df_import[site].values[:n_snap]

        if site_up in bus_base:
            # Transmission bus without generator -- inject directly
            target = bus_base[site_up]
            P[:, bus_idx[target]] += p_vals
            sub_total += p_vals
            sub_count += 1
            log.info("  tx-no-gen %-10s -> %-12s  mean=%.1f MW",
                     site_up, target, float(p_vals.mean()))
        else:
            # Sub-transmission -- aggregate to nearest AC bus
            if site_up not in node_coords:
                continue
            lat1, lon1 = node_coords[site_up]
            best_bus, best_dist = None, float("inf")
            for bus in ac_buses:
                base   = bus.rsplit("_", 1)[0] if "_" in bus else bus
                coords = node_coords.get(bus) or node_coords.get(base)
                if not coords:
                    continue
                d = _math.sqrt((coords[0]-lat1)**2 + (coords[1]-lon1)**2)
                if d < best_dist:
                    best_dist = d
                    best_bus  = bus
            if best_bus and best_bus in bus_idx:
                P[:, bus_idx[best_bus]] += p_vals
                sub_total += p_vals
                sub_count += 1
                log.info("  sub-tx %-10s -> %-12s  mean=%.1f MW",
                         site_up, best_bus, float(p_vals.mean()))

    log.info("  Additional SI GIPs: %d sites  mean=%.1f MW",
             sub_count, sub_total.mean())

    # HVDC injection at BEN
    ben_bus = next((b for b in ac_buses if b.startswith("BEN")
                    and not b.endswith("DC")), None)
    if ben_bus and ben_bus in bus_idx:
        P[:, bus_idx[ben_bus]] += hvdc.values
        log.info("  Mean HVDC injection at %s: %.1f MW",
                 ben_bus, hvdc.mean())

    # Load withdrawals -- read directly from CSV to bypass index issues
    pset_path = network_dir / "loads-p_set.csv"
    if pset_path.exists():
        load_p = pd.read_csv(str(pset_path), index_col=0)
        load_p.index = range(len(load_p))  # ensure integer index
        log.info("  loads-p_set.csv: %d rows x %d cols  mean=%.1f MW",
                 len(load_p), len(load_p.columns),
                 load_p.sum(axis=1).mean())
    else:
        load_p = n.loads_t.p_set

    load_total = np.zeros(n_snap)
    for load in n.loads.index:
        bus = n.loads.loc[load, "bus"]
        if bus not in bus_idx:
            continue
        idx = bus_idx[bus]
        if load in load_p.columns:
            vals = load_p[load].values[:n_snap]
        else:
            vals = np.full(n_snap, float(n.loads.loc[load, "p_set"]))
        P[:, idx]  -= vals
        load_total += vals

    log.info("  Mean total load: %.1f MW", load_total.mean())
    log.info("  Mean net injection (should be ~0): %.2f MW",
             P.sum(axis=1).mean())

    return P


# =============================================================================
# SOLVE DC POWER FLOW
# =============================================================================

def solve_dc_pf(B: np.ndarray,
                P: np.ndarray,
                slack_idx: int) -> np.ndarray:
    """
    Solve B * theta = P with slack bus angle fixed at zero.

    Removes slack row/col, solves reduced system, reinserts zero angle.
    Returns theta (snapshots x buses) in radians.
    """
    n_buses = B.shape[0]
    n_snap  = P.shape[0]
    keep    = [i for i in range(n_buses) if i != slack_idx]

    B_red = B[np.ix_(keep, keep)]
    P_red = P[:, keep]

    # Check condition number
    cond = np.linalg.cond(B_red)
    log.info("  B_red condition number: %.2e", cond)
    if cond > 1e10:
        log.warning("  B matrix is near-singular -- check network connectivity")

    # Solve
    theta_red = np.linalg.solve(B_red, P_red.T).T  # (n_snap, n_buses-1)

    # Reassemble with slack angle = 0
    theta = np.zeros((n_snap, n_buses))
    for j_new, j_old in enumerate(keep):
        theta[:, j_old] = theta_red[:, j_new]

    max_deg = float(np.degrees(np.abs(theta).max()))
    log.info("  Max voltage angle: %.4f rad (%.2f deg)", 
             float(np.abs(theta).max()), max_deg)
    if max_deg > 30:
        log.warning("  Large voltage angles (>30 deg) -- DC approximation may be inaccurate")

    return theta


# =============================================================================
# COMPUTE LINE FLOWS
# =============================================================================

def compute_line_flows(n: pypsa.Network,
                       theta: np.ndarray,
                       bus_idx: dict) -> pd.DataFrame:
    """
    Compute line flows from voltage angles.
    p_line = (theta_i - theta_j) / x_pu  (MW, on system base)
    """
    flows = {}
    for line in n.lines.index:
        b0   = n.lines.loc[line, "bus0"]
        b1   = n.lines.loc[line, "bus1"]
        x_pu = float(n.lines.loc[line, "x_pu_eff"]
                     if "x_pu_eff" in n.lines.columns
                     else n.lines.loc[line, "x_pu"])
        if x_pu <= 0 or b0 not in bus_idx or b1 not in bus_idx:
            continue
        i, j  = bus_idx[b0], bus_idx[b1]
        flows[line] = pd.Series(
            (theta[:, i] - theta[:, j]) / x_pu,
            index=n.snapshots
        )

    p0 = pd.DataFrame(flows)
    log.info("  Line flows computed: %d / %d lines",
             len(p0.columns), len(n.lines))
    return p0


# =============================================================================
# ANALYSIS AND OUTPUT
# =============================================================================

def analyse_and_save(n: pypsa.Network,
                     p0: pd.DataFrame,
                     P: np.ndarray,
                     ac_buses: list,
                     bus_idx: dict,
                     out_dir: Path,
                     month_label: str) -> dict:
    """Compute loading statistics and save outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Line loading
    line_loading = pd.DataFrame({
        "s_nom_MW":      n.lines.loc[p0.columns, "s_nom"],
        "max_flow_MW":   p0.abs().max(),
        "mean_flow_MW":  p0.abs().mean(),
        "p95_flow_MW":   p0.abs().quantile(0.95),
    })
    line_loading["max_loading_pct"] = (
        line_loading["max_flow_MW"] /
        line_loading["s_nom_MW"].replace(0, np.nan) * 100
    ).round(1)
    line_loading["mean_loading_pct"] = (
        line_loading["mean_flow_MW"] /
        line_loading["s_nom_MW"].replace(0, np.nan) * 100
    ).round(1)
    line_loading["p95_loading_pct"] = (
        line_loading["p95_flow_MW"] /
        line_loading["s_nom_MW"].replace(0, np.nan) * 100
    ).round(1)
    line_loading = line_loading.sort_values("max_loading_pct", ascending=False)

    overloaded = line_loading[line_loading["max_loading_pct"] > 100]

    log.info("\n  Line loading summary:")
    log.info("    Lines computed    : %d", len(line_loading))
    log.info("    Lines > 100%%     : %d", len(overloaded))
    log.info("    Top 10 most loaded:")
    for ln, row in line_loading.head(10).iterrows():
        log.info("    %-8s s_nom=%.0f MW  max=%.1f%%  mean=%.1f%%  p95=%.1f%%",
                 ln,
                 row["s_nom_MW"],
                 row["max_loading_pct"],
                 row["mean_loading_pct"],
                 row["p95_loading_pct"])

    # Bus injections (mean)
    bus_inj = pd.DataFrame({
        "bus":            ac_buses,
        "mean_inj_MW":    P.mean(axis=0),
    }).set_index("bus").sort_values("mean_inj_MW")

    # Save CSVs
    line_loading.to_csv(out_dir / "dc_line_loading.csv")
    p0.to_csv(out_dir / "dc_line_flows.csv")
    bus_inj.to_csv(out_dir / "dc_bus_injections.csv")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Line loading bar chart
    lv = line_loading["max_loading_pct"].dropna()
    lv = lv[np.isfinite(lv)]
    colors = ["red" if v > 100 else "orange" if v > 75
              else "steelblue" for v in lv.values]
    axes[0].barh(range(len(lv)), lv.values, color=colors)
    axes[0].set_yticks(range(len(lv)))
    axes[0].set_yticklabels(lv.index.tolist(), fontsize=7)
    axes[0].axvline(100, color="red", linestyle="--", linewidth=1,
                    label="Rated capacity")
    axes[0].axvline(75, color="orange", linestyle="--", linewidth=1,
                    label="75% warning")
    axes[0].set_title(f"DC Power Flow -- Line Max Loading\n{month_label}")
    axes[0].set_xlabel("Max loading (% of rated capacity)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.4, axis="x")

    # Bus injection bar chart (top 20 by absolute value)
    top_buses = bus_inj.reindex(
        bus_inj["mean_inj_MW"].abs().nlargest(20).index
    )
    colors2 = ["green" if v > 0 else "blue" for v in top_buses["mean_inj_MW"]]
    axes[1].barh(range(len(top_buses)), top_buses["mean_inj_MW"].values,
                 color=colors2)
    axes[1].set_yticks(range(len(top_buses)))
    axes[1].set_yticklabels(top_buses.index.tolist(), fontsize=8)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Mean Bus Net Injection (green=gen, blue=load)")
    axes[1].set_xlabel("Mean net injection (MW)")
    axes[1].grid(True, alpha=0.4, axis="x")

    fig.tight_layout()
    fig.savefig(out_dir / "dc_power_flow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved to: %s", out_dir)

    return {
        "n_lines":         len(line_loading),
        "n_overloaded":    int(len(overloaded)),
        "max_loading_pct": float(line_loading["max_loading_pct"].max()),
        "mean_loading_pct": float(line_loading["mean_loading_pct"].mean()),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone numpy DC power flow for SI network"
    )
    parser.add_argument("--year",  type=int, default=2024)
    parser.add_argument("--month", type=int, default=1)
    args = parser.parse_args()

    _root = os.environ.get("PYPSA_NZA_ROOT", "")
    if not _root:
        log.error("PYPSA_NZA_ROOT not set.")
        sys.exit(1)

    workspace   = Path(_root) / "pypsa_nza_workspace"
    month_label = f"{MONTHS[args.month-1]}_{args.year}"
    net_dir     = (workspace / "models" / "networks" / "reference"
                   / str(args.year) / f"si_{month_label}")
    out_dir     = (workspace / "outputs" / "verification"
                   / str(args.year) / month_label)

    log.info("=" * 60)
    log.info("DC POWER FLOW  --  numpy direct solution")
    log.info("Solving B * theta = P")
    log.info("=" * 60)
    log.info("Network : %s", net_dir)
    log.info("Month   : %s", month_label)

    if not net_dir.exists():
        log.error("Network not found: %s", net_dir)
        sys.exit(1)

    # Load network
    log.info("\nLoading network...")
    n = load_network(net_dir)

    # Build bus index (AC buses only)
    ac_buses = [b for b in n.buses.index
                if n.buses.loc[b, "carrier"] == "AC"]
    bus_idx  = {bus: i for i, bus in enumerate(ac_buses)}

    # Identify slack bus
    slack_bus = next(
        (n.generators.loc[g, "bus"]
         for g in n.generators.index
         if n.generators.loc[g, "bus"].startswith("BEN")
         and n.generators.loc[g, "bus"] in bus_idx),
        ac_buses[0]
    )
    slack_idx = bus_idx[slack_bus]
    log.info("  Slack bus: %s (index %d)", slack_bus, slack_idx)
    log.info("  AC buses : %d", len(ac_buses))

    # Load timeseries
    log.info("\nLoading timeseries...")
    df_import, hvdc = load_timeseries(workspace, args.year,
                                      args.month, n.snapshots)

    # Build B matrix
    log.info("\nBuilding susceptance matrix...")
    B = build_B_matrix(n, ac_buses, bus_idx)

    # Build node coordinates for sub-transmission aggregation
    import math as _math
    node_coords = {}
    nodes_path  = workspace / "data" / "manual" / "nodes.csv"
    if nodes_path.exists():
        df_nodes = pd.read_csv(str(nodes_path), dtype=str)
        for _, row in df_nodes.iterrows():
            site = str(row.get("site", "")).strip().upper()
            try:
                lat = float(row.get("lat", ""))
                lon = float(row.get("long", ""))
                if not (_math.isnan(lat) or _math.isnan(lon)):
                    node_coords[site] = (lat, lon)
            except (ValueError, TypeError):
                pass
        log.info("  Node coordinates loaded: %d sites", len(node_coords))

    # Build injection matrix
    log.info("\nBuilding injection matrix...")
    P = build_injection_matrix(n, df_import, hvdc, ac_buses, bus_idx,
                               net_dir, node_coords)

    # Solve DC power flow
    log.info("\nSolving DC power flow...")
    theta = solve_dc_pf(B, P, slack_idx)

    # Compute line flows
    log.info("\nComputing line flows...")
    p0 = compute_line_flows(n, theta, bus_idx)

    # Analyse and save
    log.info("\nAnalysing results...")
    results = analyse_and_save(n, p0, P, ac_buses, bus_idx,
                               out_dir, month_label)

    log.info("\n%s", "=" * 60)
    log.info("DC POWER FLOW COMPLETE")
    log.info("=" * 60)
    log.info("  Lines computed    : %d", results["n_lines"])
    log.info("  Lines > 100%%     : %d", results["n_overloaded"])
    log.info("  Max line loading  : %.1f%%", results["max_loading_pct"])
    log.info("  Mean line loading : %.1f%%", results["mean_loading_pct"])

    if results["n_overloaded"] == 0:
        log.info("  RESULT: No lines overloaded -- network topology validated")
    else:
        log.warning("  RESULT: %d lines exceed rated capacity",
                    results["n_overloaded"])
        log.warning("  Check line_loading.csv for details")


if __name__ == "__main__":
    main()
