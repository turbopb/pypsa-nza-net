#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_line_lengths.py

Compute straight-line (Euclidean NZTM) distances between transmission line
bus endpoints, populate the 'length' column, and compute the 'delta' column.

For each line in lines_data_si.csv (or lines_data_ni.csv):
    length = Euclidean distance between bus0 and bus1 in NZTM (km)
           = sqrt((E2-E1)^2 + (N2-N1)^2) / 1000
    delta  = (distance - length) / distance
           = fractional excess of QGIS route over straight-line distance
             normalised to distance (always >= 0)

NZTM Euclidean distance is used in preference to haversine because:
  - All nodes including JCT_ junction points have NZTM coordinates
  - JCT_ nodes have no WGS84 lat/long in nodes.csv
  - NZTM is a conformal projection optimised for NZ; Euclidean distance
    agrees with haversine to within 0.1% for distances up to 200 km

Inputs
------
    pypsa_nza_workspace/data/manual/nodes.csv
    pypsa_nza_workspace/data/manual/lines_data_si.csv   (or _ni.csv)

Outputs
-------
    Same file as input, with 'length' and 'delta' columns populated.
    A backup of the original is written as lines_data_si_backup.csv.

Usage
-----
    python compute_line_lengths.py [--file lines_data_si.csv]

    Default file is lines_data_si.csv.  Use --file lines_data_ni.csv
    for North Island once that file is ready.

Author: Phillippe Bruneau
"""

import os
import sys
import math
import shutil
import logging
import argparse
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
# DISTANCE COMPUTATION
# =============================================================================

def nztm_distance_km(e1: float, n1: float, e2: float, n2: float) -> float:
    """
    Euclidean distance between two NZTM coordinate pairs, in kilometres.

    NZTM coordinates are in metres.  The Euclidean distance in NZTM agrees
    with the great-circle (haversine) distance to within 0.1% for distances
    up to 200 km over New Zealand, which is sufficient for transmission line
    length estimation.

    Parameters
    ----------
    e1, n1 : float
        NZTM easting and northing of point 1 (metres)
    e2, n2 : float
        NZTM easting and northing of point 2 (metres)

    Returns
    -------
    float
        Straight-line distance in kilometres
    """
    return math.sqrt((e2 - e1) ** 2 + (n2 - n1) ** 2) / 1000.0


# =============================================================================
# MAIN
# =============================================================================

def compute_line_lengths(lines_path: Path, nodes_path: Path) -> pd.DataFrame:
    """
    Compute length and delta for all lines in the given lines file.

    Parameters
    ----------
    lines_path : Path
        Path to lines_data_si.csv or lines_data_ni.csv
    nodes_path : Path
        Path to nodes.csv

    Returns
    -------
    pd.DataFrame
        Updated lines dataframe with length and delta populated
    """
    log.info("=" * 60)
    log.info("COMPUTE LINE LENGTHS")
    log.info("=" * 60)
    log.info("Lines : %s", lines_path)
    log.info("Nodes : %s", nodes_path)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df_lines = pd.read_csv(lines_path, dtype=str)
    df_nodes = pd.read_csv(nodes_path, dtype=str)

    # Drop unnamed columns created by trailing commas in the CSV header
    df_lines = df_lines.loc[:, ~df_lines.columns.str.startswith("Unnamed")]
    df_nodes = df_nodes.loc[:, ~df_nodes.columns.str.startswith("Unnamed")]

    for col in df_lines.columns:
        if df_lines[col].dtype == object:
            df_lines[col] = df_lines[col].str.strip()
    for col in df_nodes.columns:
        if df_nodes[col].dtype == object:
            df_nodes[col] = df_nodes[col].str.strip()

    df_nodes["X"] = pd.to_numeric(df_nodes["X"], errors="coerce")
    df_nodes["Y"] = pd.to_numeric(df_nodes["Y"], errors="coerce")
    df_lines["distance"] = pd.to_numeric(df_lines["distance"], errors="coerce")

    # Build NZTM coordinate lookup keyed by site code
    coord_lkp = df_nodes.set_index("site")[["X", "Y", "name"]]

    log.info("\nLoaded %d lines, %d nodes", len(df_lines), len(df_nodes))

    # ------------------------------------------------------------------
    # Compute length and delta for each line
    # ------------------------------------------------------------------
    lengths = []
    deltas  = []
    warnings = []

    for _, row in df_lines.iterrows():
        line = row["line"]
        b0   = row["bus0"].strip().upper()
        b1   = row["bus1"].strip().upper()
        dist = row["distance"]   # QGIS measured distance (may be NaN)

        # Look up NZTM coordinates
        if b0 not in coord_lkp.index:
            log.warning("  %s: bus0 '%s' not in nodes.csv -- length set to NaN",
                        line, b0)
            lengths.append(float("nan"))
            deltas.append(float("nan"))
            warnings.append(line)
            continue

        if b1 not in coord_lkp.index:
            log.warning("  %s: bus1 '%s' not in nodes.csv -- length set to NaN",
                        line, b1)
            lengths.append(float("nan"))
            deltas.append(float("nan"))
            warnings.append(line)
            continue

        e0, n0 = coord_lkp.loc[b0, "X"], coord_lkp.loc[b0, "Y"]
        e1, n1 = coord_lkp.loc[b1, "X"], coord_lkp.loc[b1, "Y"]

        if any(pd.isna(v) for v in (e0, n0, e1, n1)):
            log.warning("  %s: missing NZTM coordinates for %s or %s",
                        line, b0, b1)
            lengths.append(float("nan"))
            deltas.append(float("nan"))
            warnings.append(line)
            continue

        length = nztm_distance_km(e0, n0, e1, n1)
        lengths.append(round(length, 6))

        # Delta: fractional excess of QGIS distance over straight-line
        if pd.notna(dist) and dist > 0:
            delta = (dist - length) / dist
            deltas.append(round(delta, 6))
        else:
            deltas.append(float("nan"))

    df_lines["length"] = lengths
    df_lines["delta"]  = deltas

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_ok      = sum(1 for l in lengths if not math.isnan(l))
    n_warn    = len(warnings)
    n_dist    = df_lines["distance"].notna().sum()
    n_delta   = df_lines["delta"].notna().sum()

    log.info("\n%s", "=" * 60)
    log.info("SUMMARY")
    log.info("%s", "=" * 60)
    log.info("  Lines processed      : %d", len(df_lines))
    log.info("  Length computed      : %d", n_ok)
    log.info("  Delta computed       : %d  (requires distance)", n_delta)
    log.info("  Warnings             : %d", n_warn)

    if n_delta > 0:
        valid_delta = df_lines["delta"].dropna()
        log.info("\n  Delta statistics (distance vs straight-line):")
        log.info("    Min    : %.4f  (%.1f%% longer than straight-line)",
                 valid_delta.min(), valid_delta.min() * 100)
        log.info("    Median : %.4f  (%.1f%%)", valid_delta.median(),
                 valid_delta.median() * 100)
        log.info("    Max    : %.4f  (%.1f%%)", valid_delta.max(),
                 valid_delta.max() * 100)
        log.info("    Mean   : %.4f  (%.1f%%)", valid_delta.mean(),
                 valid_delta.mean() * 100)

        # Flag high-delta lines (route > 50% longer than straight line)
        high_delta = df_lines[df_lines["delta"] > 0.5][
            ["line","name","bus0","bus1","length","distance","delta"]
        ]
        if not high_delta.empty:
            log.info("\n  Lines where route is >50%% longer than straight-line:")
            for _, r in high_delta.iterrows():
                log.info("    %s  %s  length=%.1f km  distance=%.1f km  delta=%.3f",
                         r["line"], r["name"], r["length"],
                         r["distance"], r["delta"])

    return df_lines


def main():
    parser = argparse.ArgumentParser(
        description="Compute NZTM straight-line lengths and delta for transmission lines"
    )
    parser.add_argument(
        "--file",
        default="lines_data_si.csv",
        help="Lines CSV filename (default: lines_data_si.csv)"
    )
    args = parser.parse_args()

    # Workspace paths
    _root = os.environ.get("PYPSA_NZA_ROOT", "")
    if not _root:
        log.error("PYPSA_NZA_ROOT not set. See SETUP.md.")
        sys.exit(1)

    workspace  = Path(_root) / "pypsa_nza_workspace"
    manual_dir = workspace / "data" / "manual"

    lines_path = manual_dir / args.file
    nodes_path = manual_dir / "nodes.csv"

    if not lines_path.exists():
        log.error("Lines file not found: %s", lines_path)
        sys.exit(1)
    if not nodes_path.exists():
        log.error("Nodes file not found: %s", nodes_path)
        sys.exit(1)

    # Backup original
    backup_path = lines_path.with_name(
        lines_path.stem + "_backup" + lines_path.suffix
    )
    shutil.copy2(lines_path, backup_path)
    log.info("Backup written to: %s", backup_path)

    # Compute
    df_updated = compute_line_lengths(lines_path, nodes_path)

    # Save -- preserve all original columns, just update length and delta
    df_updated.to_csv(lines_path, index=False)
    log.info("\nUpdated file saved: %s", lines_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
