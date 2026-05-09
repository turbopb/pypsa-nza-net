#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_power_flow.py

AC power flow verification for the SI network using measured data.

Follows the workflow described in the PyPSA documentation:
1. Load network topology
2. Set loads_t.p_set from measured GXP export timeseries
3. Set generators_t.p_set from measured GIP import timeseries
4. Set generator control: PQ for all, Slack for BEN
5. Run n.pf()
6. Inspect line flows, transformer flows, losses

Usage
-----
    python verify_power_flow.py [--year 2024] [--month 1]
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
    return n


# =============================================================================
# TIMESERIES LOADING
# =============================================================================

def load_import_timeseries(annual_dir: Path,
                           year: int,
                           snapshots: pd.DatetimeIndex) -> pd.DataFrame:
    fpath = annual_dir / f"{year}_import_sites_all.csv"
    if not fpath.exists():
        log.error("Import timeseries not found: %s", fpath)
        sys.exit(1)
    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    df.index = df.index.tz_localize(None)
    return df.reindex(snapshots).fillna(0.0)


def load_hvdc_timeseries(hvdc_dir: Path,
                         year: int,
                         month: int,
                         snapshots: pd.DatetimeIndex) -> pd.Series:
    fname = f"{year}{month:02d}_dc_MW.csv"
    fpath = hvdc_dir / fname
    if not fpath.exists():
        log.warning("HVDC file not found: %s -- set to zero", fpath)
        return pd.Series(0.0, index=snapshots)
    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    df.index = df.index.tz_localize(None)
    if "BEN2201" not in df.columns:
        log.warning("BEN2201 not found -- HVDC set to zero")
        return pd.Series(0.0, index=snapshots)
    series = df["BEN2201"].reindex(snapshots).fillna(0.0)
    log.info("  HVDC BEN2201: mean=%.1f MW  min=%.1f MW  max=%.1f MW",
             series.mean(), series.min(), series.max())
    return series


# =============================================================================
# SETUP FOR PF
# =============================================================================

def setup_power_flow(n: pypsa.Network,
                     df_import: pd.DataFrame,
                     hvdc_series: pd.Series) -> pypsa.Network:
    """
    Set up network for AC power flow following the PyPSA pf() workflow:

    1. loads_t.p_set -- already set from network construction (all GXPs)
    2. generators_t.p_set -- set from measured GIP timeseries
    3. generator control -- PQ for all generators, Slack for BEN
    4. v_set_pu -- 1.0 for all generators
    5. HVDC -- added as negative load at BEN

    The Slack generator (BEN_hydro) absorbs the power balance residual.
    This residual represents sub-transmission generation not explicitly
    modelled but already netted into the GXP load timeseries.
    """
    log.info("\nSetting up network for pf()...")

    # Set all generators to PQ control with v_set_pu = 1.0
    n.generators["control"]   = "PQ"
    n.generators["v_set_pu"]  = 1.0
    n.generators["p_max_pu"]  = 1.0
    n.generators["p_min_pu"]  = 0.0

    # Set BEN_hydro as slack bus
    slack_gen = None
    for gen in n.generators.index:
        if n.generators.loc[gen, "bus"].startswith("BEN"):
            slack_gen = gen
            break
    if slack_gen is None:
        slack_gen = n.generators.index[0]
    n.generators.loc[slack_gen, "control"] = "Slack"
    log.info("  Slack bus: %s", slack_gen)

    # Group generators by site for multi-voltage equal split
    site_gens = {}
    for gen in n.generators.index:
        bus  = n.generators.loc[gen, "bus"]
        site = bus.rsplit("_", 1)[0] if "_" in bus else bus
        site_gens.setdefault(site, [])
        site_gens[site].append((gen, bus))

    # Set p_set from measured GIP data
    for site, gen_bus_list in site_gens.items():
        if site not in df_import.columns:
            log.warning("  %-20s no GIP data", site)
            for gen, _ in gen_bus_list:
                n.generators_t.p_set[gen] = pd.Series(0.0, index=n.snapshots)
            continue
        fraction = 1.0 / len(gen_bus_list)
        for gen, bus in gen_bus_list:
            p_vals = df_import[site].values * fraction
            n.generators_t.p_set[gen] = pd.Series(p_vals, index=n.snapshots)
            log.info("  %-25s  %s  <- %s*%.2f  (mean=%.1f MW)",
                     gen,
                     "Slack" if gen == slack_gen else "PQ  ",
                     site, fraction,
                     float(df_import[site].mean() * fraction))

    # HVDC: add as negative load at BEN (import = injection into SI)
    if "BEN" in n.buses.index:
        hvdc_max = float(hvdc_series.max())
        if hvdc_max > 0:
            lname = "inj_HVDC_BEN"
            if lname not in n.loads.index:
                n.add("Load", lname, bus="BEN", p_set=0.0)
            n.loads_t.p_set[lname] = pd.Series(
                -hvdc_series.values, index=n.snapshots)
            log.info("  HVDC at BEN: mean=%.1f MW (negative load)",
                     hvdc_series.mean())

    # Set bus voltage nominal for proper per-unit base
    n.buses["v_nom"] = n.buses["v_nom"].where(n.buses["v_nom"] > 0, 110.0)

    # Log expected balance
    gen_mean = sum(
        float(df_import[site].mean())
        for site in site_gens
        if site in df_import.columns
    )
    log.info("  Expected generation (ex slack): %.1f MW", gen_mean)
    log.info("  HVDC: %.1f MW", hvdc_series.mean())
    log.info("  Total load (all GXPs): %.1f MW",
             n.loads_t.p_set.sum(axis=1).mean()
             + hvdc_series.mean())

    return n


# =============================================================================
# POWER FLOW AND ANALYSIS
# =============================================================================

def run_power_flow(n: pypsa.Network,
                   out_dir: Path,
                   month_label: str) -> dict:
    """
    Run AC power flow and analyse results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("\nRunning AC power flow (n.pf())...")
    pf_results = n.pf(use_seed=True)

    # Check convergence
    if not pf_results["converged"].all(axis=None):
        n_failed = (~pf_results["converged"]).sum().sum()
        log.warning("  Power flow did not converge for %d snapshots", n_failed)
    else:
        log.info("  Power flow converged for all snapshots")

    # ------------------------------------------------------------------
    # Line loading
    # ------------------------------------------------------------------
    line_loading = pd.DataFrame()
    if not n.lines_t.p0.empty:
        p0 = n.lines_t.p0.copy()
        valid_lines = p0.columns[p0.notna().all(axis=0)]
        nan_lines   = p0.columns[~p0.columns.isin(valid_lines)]
        log.info("  Lines with valid flows: %d / %d",
                 len(valid_lines), len(p0.columns))
        if len(nan_lines) > 0:
            log.warning("  Lines with NaN flows: %d", len(nan_lines))

        if len(valid_lines) > 0:
            line_loading = pd.DataFrame({
                "s_nom_MW":      n.lines.loc[valid_lines, "s_nom"],
                "max_flow_MW":   p0[valid_lines].abs().max(),
                "mean_flow_MW":  p0[valid_lines].abs().mean(),
                "p95_flow_MW":   p0[valid_lines].abs().quantile(0.95),
            })
            line_loading["max_loading_pct"] = (
                line_loading["max_flow_MW"] /
                line_loading["s_nom_MW"].replace(0, np.nan) * 100
            ).round(1)
            line_loading["mean_loading_pct"] = (
                line_loading["mean_flow_MW"] /
                line_loading["s_nom_MW"].replace(0, np.nan) * 100
            ).round(1)
            line_loading = line_loading.sort_values("max_loading_pct",
                                                    ascending=False)
            line_loading.to_csv(out_dir / "line_loading.csv")

            overloaded = line_loading[line_loading["max_loading_pct"] > 100]
            log.info("\n  Line loading:")
            log.info("    Lines with max loading > 100%%: %d", len(overloaded))
            log.info("    Top 10 most loaded lines:")
            for ln, row in line_loading.head(10).iterrows():
                log.info("    %-8s s_nom=%.0f MW  max=%.1f%%  mean=%.1f%%",
                         ln, row["s_nom_MW"],
                         row["max_loading_pct"],
                         row["mean_loading_pct"])

    # ------------------------------------------------------------------
    # Generation results
    # ------------------------------------------------------------------
    gen_p = n.generators_t.p
    inj_cols  = [c for c in n.loads.index if c.startswith("inj_")]
    load_cols = [c for c in n.loads.index if not c.startswith("inj_")]

    total_gen    = gen_p.sum(axis=1) if not gen_p.empty else \
                   pd.Series(0.0, index=n.snapshots)
    total_demand = n.loads_t.p_set[load_cols].sum(axis=1) \
                   if load_cols else pd.Series(0.0, index=n.snapshots)
    hvdc_cols    = [c for c in inj_cols if "HVDC" in c]
    hvdc_gen     = -n.loads_t.p_set[hvdc_cols].sum(axis=1) \
                   if hvdc_cols else pd.Series(0.0, index=n.snapshots)

    log.info("\n  Generation results:")
    log.info("    Mean total generation: %.1f MW", total_gen.mean())
    log.info("    Mean HVDC import     : %.1f MW", hvdc_gen.mean())
    log.info("    Mean total demand    : %.1f MW", total_demand.mean())

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    t = range(len(n.snapshots))
    axes[0].plot(t, total_demand.values,
                 label="Total demand (GXPs)", color="blue", linewidth=0.8)
    axes[0].plot(t, total_gen.values,
                 label="Total generation (solved)", color="green",
                 linewidth=0.8, linestyle="--")
    axes[0].plot(t, hvdc_gen.values,
                 label="HVDC import (measured)", color="red",
                 linewidth=0.8, alpha=0.7)
    axes[0].set_title(f"SI Network Power Balance -- {month_label}")
    axes[0].set_xlabel("Timestep (30 min)")
    axes[0].set_ylabel("MW")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.4)

    if not line_loading.empty:
        lv = line_loading["max_loading_pct"].dropna()
        lv = lv[np.isfinite(lv)]
        if len(lv) > 0:
            axes[1].barh(range(len(lv)), lv.values,
                         color=["red" if v > 100 else "steelblue"
                                for v in lv.values])
            axes[1].set_yticks(range(len(lv)))
            axes[1].set_yticklabels(lv.index.tolist(), fontsize=6)
            axes[1].axvline(100, color="red", linestyle="--",
                            linewidth=1, label="Rated capacity")
            axes[1].set_title("Line Maximum Loading (% of rated capacity)")
            axes[1].set_xlabel("Max loading (%)")
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.4, axis="x")

    fig.tight_layout()
    fig.savefig(out_dir / "power_flow_verification.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("\n  Figures saved to: %s", out_dir)

    n_overloaded = int(len(overloaded)) if not line_loading.empty else 0
    max_loading  = float(line_loading["max_loading_pct"].max()) \
                   if not line_loading.empty else 0.0

    return {
        "mean_demand_MW":  float(total_demand.mean()),
        "mean_gen_MW":     float(total_gen.mean()),
        "mean_hvdc_MW":    float(hvdc_gen.mean()),
        "n_overloaded":    n_overloaded,
        "max_loading_pct": max_loading,
        "n_valid_lines":   len(valid_lines) if not n.lines_t.p0.empty else 0,
    }


# =============================================================================
# NUMPY DC POWER FLOW
# =============================================================================

def run_dc_power_flow(n: pypsa.Network,
                      df_import: pd.DataFrame,
                      hvdc_series: pd.Series,
                      out_dir: Path,
                      month_label: str) -> dict:
    """
    DC power flow solved directly using numpy.

    This bypasses PyPSA's lpf() API entirely and solves the DC power
    flow equations directly:

        B * theta = P

    where:
        B     = bus susceptance matrix (1/x for each line)
        theta = bus voltage angles (radians)
        P     = net injection at each bus (generation - load, MW)

    Line flows are then computed as:
        p_line = (theta_i - theta_j) / x_ij  * s_base

    The slack bus (BEN) has its angle fixed at zero.

    This is a baseline check -- it makes no assumptions about PyPSA's
    internal API and confirms the network topology and impedances are
    consistent with the measured injection data.
    """
    import math as _math

    log.info("\nRunning numpy DC power flow (baseline check)...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build bus index
    # ------------------------------------------------------------------
    # Use only AC buses in the main connected component
    ac_buses  = n.buses[n.buses["carrier"] == "AC"].index.tolist()
    bus_idx   = {bus: i for i, bus in enumerate(ac_buses)}
    n_buses   = len(ac_buses)

    # Identify slack bus
    slack_bus = None
    for gen in n.generators.index:
        bus = n.generators.loc[gen, "bus"]
        if bus.startswith("BEN") and bus in bus_idx:
            slack_bus = bus
            break
    if slack_bus is None:
        slack_bus = ac_buses[0]
    slack_idx = bus_idx[slack_bus]
    log.info("  Slack bus: %s (index %d)", slack_bus, slack_idx)
    log.info("  AC buses: %d", n_buses)

    # ------------------------------------------------------------------
    # 2. Build susceptance matrix B
    # ------------------------------------------------------------------
    B = np.zeros((n_buses, n_buses))

    # AC lines
    for line in n.lines.index:
        b0   = n.lines.loc[line, "bus0"]
        b1   = n.lines.loc[line, "bus1"]
        x_pu = float(n.lines.loc[line, "x_pu"])
        if x_pu <= 0 or b0 not in bus_idx or b1 not in bus_idx:
            continue
        b = 1.0 / x_pu
        i, j = bus_idx[b0], bus_idx[b1]
        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b

    # Transformers (treated as lines with x_pu from transformer data)
    for trafo in n.transformers.index:
        b0   = n.transformers.loc[trafo, "bus0"]
        b1   = n.transformers.loc[trafo, "bus1"]
        x_pu = float(n.transformers.loc[trafo, "x_pu"])
        if x_pu <= 0 or b0 not in bus_idx or b1 not in bus_idx:
            continue
        b = 1.0 / x_pu
        i, j = bus_idx[b0], bus_idx[b1]
        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b

    log.info("  B matrix: %dx%d  non-zero off-diagonal: %d",
             n_buses, n_buses, int((B != 0).sum()) - n_buses)

    # ------------------------------------------------------------------
    # 3. Build net injection matrix P (snapshots x buses)
    # ------------------------------------------------------------------
    n_snap = len(n.snapshots)
    P      = np.zeros((n_snap, n_buses))

    # Generation injections from GIP timeseries
    site_gens = {}
    for gen in n.generators.index:
        bus  = n.generators.loc[gen, "bus"]
        site = bus.rsplit("_", 1)[0] if "_" in bus else bus
        site_gens.setdefault(site, [])
        site_gens[site].append((gen, bus))

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

    # HVDC injection at BEN
    if slack_bus in bus_idx:
        P[:, bus_idx[slack_bus]] += hvdc_series.values

    # Load withdrawals (positive p_set = demand)
    load_p_set = n.loads_t.p_set.reindex(n.snapshots).fillna(0.0)
    for load in n.loads.index:
        bus = n.loads.loc[load, "bus"]
        if bus not in bus_idx:
            continue
        idx = bus_idx[bus]
        if load in load_p_set.columns:
            P[:, idx] -= load_p_set[load].values
        else:
            p_static = float(n.loads.loc[load, "p_set"])
            P[:, idx] -= p_static

    log.info("  Mean net injection (should be ~0): %.2f MW",
             P.mean(axis=1).mean())

    # ------------------------------------------------------------------
    # 4. Solve B_red * theta = P_red  (remove slack row/col)
    # ------------------------------------------------------------------
    keep = [i for i in range(n_buses) if i != slack_idx]
    B_red = B[np.ix_(keep, keep)]
    P_red = P[:, keep]

    try:
        theta_red = np.linalg.solve(B_red, P_red.T).T  # (n_snap, n_buses-1)
    except np.linalg.LinAlgError as e:
        log.error("  DC power flow singular: %s", e)
        return {"status": "singular"}

    # Reassemble full angle vector
    theta = np.zeros((n_snap, n_buses))
    for j_new, j_old in enumerate(keep):
        theta[:, j_old] = theta_red[:, j_new]

    log.info("  Max voltage angle: %.4f rad (%.2f deg)",
             float(np.abs(theta).max()),
             float(np.degrees(np.abs(theta).max())))

    # ------------------------------------------------------------------
    # 5. Compute line flows
    # ------------------------------------------------------------------
    line_results = {}
    for line in n.lines.index:
        b0   = n.lines.loc[line, "bus0"]
        b1   = n.lines.loc[line, "bus1"]
        x_pu = float(n.lines.loc[line, "x_pu"])
        if x_pu <= 0 or b0 not in bus_idx or b1 not in bus_idx:
            continue
        i, j  = bus_idx[b0], bus_idx[b1]
        p_flow = (theta[:, i] - theta[:, j]) / x_pu  # in per-unit
        line_results[line] = pd.Series(p_flow, index=n.snapshots)

    p0_dc = pd.DataFrame(line_results)
    log.info("  Lines with computed flows: %d / %d",
             len(p0_dc.columns), len(n.lines))

    # ------------------------------------------------------------------
    # 6. Line loading analysis
    # ------------------------------------------------------------------
    line_loading_dc = pd.DataFrame({
        "s_nom_MW":      n.lines.loc[p0_dc.columns, "s_nom"],
        "max_flow_MW":   p0_dc.abs().max(),
        "mean_flow_MW":  p0_dc.abs().mean(),
        "p95_flow_MW":   p0_dc.abs().quantile(0.95),
    })
    line_loading_dc["max_loading_pct"] = (
        line_loading_dc["max_flow_MW"] /
        line_loading_dc["s_nom_MW"].replace(0, np.nan) * 100
    ).round(1)
    line_loading_dc["mean_loading_pct"] = (
        line_loading_dc["mean_flow_MW"] /
        line_loading_dc["s_nom_MW"].replace(0, np.nan) * 100
    ).round(1)
    line_loading_dc = line_loading_dc.sort_values("max_loading_pct",
                                                   ascending=False)
    line_loading_dc.to_csv(out_dir / "line_loading_dc.csv")

    overloaded = line_loading_dc[line_loading_dc["max_loading_pct"] > 100]
    log.info("\n  DC Power Flow -- Line loading:")
    log.info("    Lines with max loading > 100%%: %d", len(overloaded))
    log.info("    Top 10 most loaded lines:")
    for ln, row in line_loading_dc.head(10).iterrows():
        log.info("    %-8s s_nom=%.0f MW  max=%.1f%%  mean=%.1f%%",
                 ln, row["s_nom_MW"],
                 row["max_loading_pct"],
                 row["mean_loading_pct"])

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    lv = line_loading_dc["max_loading_pct"].dropna()
    lv = lv[np.isfinite(lv)]
    ax.barh(range(len(lv)), lv.values,
            color=["red" if v > 100 else "steelblue" for v in lv.values])
    ax.set_yticks(range(len(lv)))
    ax.set_yticklabels(lv.index.tolist(), fontsize=6)
    ax.axvline(100, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"DC Power Flow -- Line Maximum Loading ({month_label})")
    ax.set_xlabel("Max loading (% of rated capacity)")
    ax.grid(True, alpha=0.4, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "dc_power_flow_verification.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "status":          "ok",
        "n_lines":         len(p0_dc.columns),
        "n_overloaded":    int(len(overloaded)),
        "max_loading_pct": float(line_loading_dc["max_loading_pct"].max()),
        "p0_dc":           p0_dc,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AC power flow verification using measured injection data"
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
    annual_dir  = workspace / "data" / "processed" / "annual"
    hvdc_dir    = (workspace / "data" / "processed"
                   / str(args.year) / "hvdc" / "dc_MW")
    out_dir     = (workspace / "outputs" / "verification"
                   / str(args.year) / month_label)

    log.info("=" * 60)
    log.info("AC POWER FLOW VERIFICATION")
    log.info("=" * 60)
    log.info("Network : %s", net_dir)
    log.info("Month   : %s", month_label)

    if not net_dir.exists():
        log.error("Network not found: %s", net_dir)
        sys.exit(1)

    n = load_network(net_dir)
    log.info("\nNetwork loaded:")
    log.info("  Buses: %d  Lines: %d  Transformers: %d  "
             "Generators: %d  Loads: %d",
             len(n.buses), len(n.lines), len(n.transformers),
             len(n.generators), len(n.loads))

    log.info("\nLoading measured timeseries...")
    df_import   = load_import_timeseries(annual_dir, args.year, n.snapshots)
    hvdc_series = load_hvdc_timeseries(hvdc_dir, args.year,
                                       args.month, n.snapshots)

    n = setup_power_flow(n, df_import, hvdc_series)

    results = run_power_flow(n, out_dir, month_label)

    # DC power flow baseline check
    log.info("\n%s", "=" * 60)
    log.info("DC POWER FLOW BASELINE (numpy)")
    log.info("%s", "=" * 60)
    dc_results = run_dc_power_flow(n, df_import, hvdc_series,
                                   out_dir, month_label)

    log.info("\n%s", "=" * 60)
    log.info("VERIFICATION COMPLETE")
    log.info("%s", "=" * 60)
    log.info("  Mean demand         : %.1f MW", results["mean_demand_MW"])
    log.info("  Mean generation     : %.1f MW", results["mean_gen_MW"])
    log.info("  Mean HVDC import    : %.1f MW", results["mean_hvdc_MW"])
    log.info("  Valid line flows    : %d / %d",
             results["n_valid_lines"], len(n.lines))
    log.info("  Overloaded lines    : %d", results["n_overloaded"])
    log.info("  Max line loading    : %.1f%%", results["max_loading_pct"])

    if results["n_valid_lines"] == len(n.lines):
        log.info("  RESULT: All line flows computed successfully")
    else:
        log.warning("  RESULT: Some line flows are NaN -- check convergence")

    if results["n_overloaded"] == 0:
        log.info("  RESULT: No lines overloaded")
    else:
        log.warning("  RESULT: %d lines exceed rated capacity",
                    results["n_overloaded"])

    if dc_results.get("status") == "ok":
        log.info("\n  DC baseline:")
        log.info("    Valid line flows : %d / %d",
                 dc_results["n_lines"], len(n.lines))
        log.info("    Overloaded lines : %d",
                 dc_results["n_overloaded"])
        log.info("    Max line loading : %.1f%%",
                 dc_results["max_loading_pct"])


if __name__ == "__main__":
    main()
