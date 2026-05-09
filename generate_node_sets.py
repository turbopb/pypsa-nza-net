#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_node_sets.py

Derive three node set CSV files from nodes.csv and lines_data_si.csv:

    network_nodes.csv
        Sites that are endpoints of at least one transmission line at
        >= 110 kV.  These are the buses explicitly represented in the
        PyPSA network topology.

    participating_nodes.csv
        Network nodes PLUS load-only sites whose demand appears in the
        EA annual timeseries (grid export file).  These sites participate
        in the model either as topology buses or as loads.

    master_nodes.csv
        All registered Transpower sites in nodes.csv, regardless of
        model participation.  Includes ghost sites and sub-transmission
        sites.  Identical to nodes.csv but with participation flags added.

Each output file includes all columns from nodes.csv plus:
    in_network      : bool -- site is a line endpoint at >= 110 kV
    in_timeseries   : bool -- site appears in EA export timeseries
    participation   : str  -- 'network', 'load_only', or 'ghost'

Outputs written to:
    pypsa_nza_workspace/data/processed/static/

Usage
-----
    python generate_node_sets.py [--year 2024]

Author: Phillippe Bruneau
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd


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
# CONSTANTS
# =============================================================================

AC_VOLT_THRESHOLD = 110   # kV -- model boundary
DC_SITES          = {"BEN-DC", "FTB", "OTB", "HAY-DC"}
ELECTRODE_SITES   = {"BGR", "THW"}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate network, participating, and master node sets"
    )
    parser.add_argument(
        "--year", type=int, default=2024,
        help="Year for EA timeseries lookup (default: 2024)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Workspace paths
    # ------------------------------------------------------------------
    _root = os.environ.get("PYPSA_NZA_ROOT", "")
    if not _root:
        log.error("PYPSA_NZA_ROOT not set. See SETUP.md.")
        sys.exit(1)

    workspace  = Path(_root) / "pypsa_nza_workspace"
    manual_dir = workspace / "data" / "manual"
    annual_dir = workspace / "data" / "processed" / "annual"
    out_dir    = workspace / "data" / "processed" / "static"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load nodes.csv (master set)
    # ------------------------------------------------------------------
    nodes_path = manual_dir / "nodes.csv"
    if not nodes_path.exists():
        log.error("nodes.csv not found: %s", nodes_path)
        sys.exit(1)

    df_nodes = pd.read_csv(nodes_path, dtype=str)
    for col in df_nodes.columns:
        if df_nodes[col].dtype == object:
            df_nodes[col] = df_nodes[col].str.strip()
    df_nodes["volts_num"] = pd.to_numeric(df_nodes["volts"], errors="coerce")

    log.info("Master node set: %d sites", len(df_nodes))

    # ------------------------------------------------------------------
    # Derive network node set from lines files
    # ------------------------------------------------------------------
    # Use all available island-specific and combined lines files
    lines_files = list(manual_dir.glob("lines_data*.csv"))
    lines_files = [f for f in lines_files if "backup" not in f.name.lower()
                   and "old" not in f.name.lower()]

    log.info("Lines files found: %s", [f.name for f in lines_files])

    transmission_buses = set()
    for lines_path in lines_files:
        df_lines = pd.read_csv(lines_path, dtype=str)
        for col in df_lines.columns:
            if df_lines[col].dtype == object:
                df_lines[col] = df_lines[col].str.strip()
        df_lines["volts_num"] = pd.to_numeric(df_lines["volts"], errors="coerce")

        df_ac = df_lines[
            df_lines["volts_num"].notna() &
            (df_lines["volts_num"] >= AC_VOLT_THRESHOLD) &
            (df_lines["volts_num"] < 350)
        ].copy()

        for col in ("bus0", "bus1"):
            for site in df_ac[col].str.upper():
                # Strip voltage suffix if present (e.g. ROX_220 -> ROX)
                base = site.rsplit("_", 1)[0] if "_" in site else site
                transmission_buses.add(base)
                transmission_buses.add(site)

    log.info("Transmission bus codes from lines: %d", len(transmission_buses))

    # ------------------------------------------------------------------
    # Load EA export timeseries to find participating load sites
    # ------------------------------------------------------------------
    export_path = annual_dir / f"{args.year}_export_sites_all.csv"
    if export_path.exists():
        df_export = pd.read_csv(export_path, nrows=1)
        timeseries_sites = set(
            col.strip().upper() for col in df_export.columns
            if col.strip().upper() not in ("TIMESTAMP", "DATETIME", "DATE", "TIME")
        )
        log.info("EA export timeseries sites (%d): year=%d",
                 len(timeseries_sites), args.year)
    else:
        log.warning("Export timeseries not found: %s", export_path)
        log.warning("Participating set will equal network set")
        timeseries_sites = set()

    # ------------------------------------------------------------------
    # Assign participation flags
    # ------------------------------------------------------------------
    def classify(site):
        s = site.strip().upper()
        base = s.rsplit("_", 1)[0] if "_" in s else s
        in_net = (s in transmission_buses) or (base in transmission_buses)
        in_ts  = (s in timeseries_sites) or (base in timeseries_sites)

        if in_net:
            return True, in_ts, "network"
        elif in_ts:
            return False, True, "load_only"
        else:
            return False, False, "ghost"

    results = df_nodes["site"].apply(classify)
    df_nodes["in_network"]    = results.apply(lambda t: t[0])
    df_nodes["in_timeseries"] = results.apply(lambda t: t[1])
    df_nodes["participation"] = results.apply(lambda t: t[2])

    # Also flag DC and electrode sites
    df_nodes.loc[df_nodes["site"].isin(DC_SITES),       "participation"] = "dc_converter"
    df_nodes.loc[df_nodes["site"].isin(ELECTRODE_SITES),"participation"] = "electrode"

    # ------------------------------------------------------------------
    # Write three output files
    # ------------------------------------------------------------------

    # 1. Master set -- all nodes with flags
    master_path = out_dir / "master_nodes.csv"
    df_nodes.to_csv(master_path, index=False)
    log.info("master_nodes.csv : %d sites", len(df_nodes))

    # 2. Network nodes -- line endpoints at >= 110 kV
    df_network = df_nodes[df_nodes["in_network"]].copy()
    network_path = out_dir / "network_nodes.csv"
    df_network.to_csv(network_path, index=False)
    log.info("network_nodes.csv: %d sites", len(df_network))

    # 3. Participating nodes -- network + load_only
    df_participating = df_nodes[
        df_nodes["participation"].isin(["network", "load_only", "dc_converter"])
    ].copy()
    participating_path = out_dir / "participating_nodes.csv"
    df_participating.to_csv(participating_path, index=False)
    log.info("participating_nodes.csv: %d sites", len(df_participating))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    counts = df_nodes["participation"].value_counts()
    log.info("\n%s", "=" * 60)
    log.info("NODE SET SUMMARY")
    log.info("%s", "=" * 60)
    for category, count in counts.items():
        log.info("  %-20s : %d", category, count)
    log.info("  %s", "-" * 40)
    log.info("  %-20s : %d", "TOTAL", len(df_nodes))

    log.info("\nOutputs written to: %s", out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
