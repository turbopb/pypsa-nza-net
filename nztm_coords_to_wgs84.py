#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nztm_coords_to_wgs84.py

Populate WGS84 lat/long coordinates in nodes.csv for any node that has
NZTM easting/northing (X, Y) but missing lat/long values.

Nodes that already have both lat and long are left unchanged.
A backup of the original file is written before any changes are made.

Uses the nztm_geod.py NZTM-to-WGS84 converter (Phillippe Bruneau, 2024).

Inputs
------
    pypsa_nza_workspace/data/manual/nodes.csv

Outputs
-------
    Same file updated in-place.
    Backup written to nodes_backup.csv.

Usage
-----
    python nztm_coords_to_wgs84.py

Author: Phillippe Bruneau / pypsa-nza-net
"""

import os
import sys
import shutil
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from nztm_geod import TransverseMercator, init_tm_proj, tm_geod
except ImportError:
    print("ERROR: nztm_geod.py not found. Place it in the same directory.")
    sys.exit(1)


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
# CONVERSION
# =============================================================================

def convert_nztm_to_wgs84(df: pd.DataFrame, tm) -> pd.DataFrame:
    """
    Convert NZTM (X, Y) to WGS84 (long, lat) for all rows where
    X and Y are present but lat or long is missing.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns X, Y, lat, long.
    tm : TransverseMercator
        Initialised projection instance.

    Returns
    -------
    pd.DataFrame
        Updated dataframe with lat and long populated.
    """
    mask = (
        df["X"].notna() &
        df["Y"].notna() &
        (df["long"].isna() | df["lat"].isna())
    )
    candidates = df[mask]
    log.info("Nodes with NZTM but missing WGS84: %d / %d",
             len(candidates), len(df))

    updated = 0
    for idx, row in candidates.iterrows():
        try:
            lat_rad, lon_rad = tm_geod(tm, row["X"], row["Y"])
            lat_deg = float(np.rad2deg(lat_rad))
            lon_deg = float(np.rad2deg(lon_rad))

            # Sanity check against NZ bounding box
            if not (-48.0 < lat_deg < -34.0 and 166.0 < lon_deg < 178.0):
                log.warning("  %-15s out-of-range result (%.4f, %.4f) -- skipped",
                            row["site"], lat_deg, lon_deg)
                continue

            df.at[idx, "lat"]  = round(lat_deg, 6)
            df.at[idx, "long"] = round(lon_deg, 6)
            log.info("  %-15s (%.1f, %.1f) -> (%.6f, %.6f)",
                     row["site"], row["X"], row["Y"], lat_deg, lon_deg)
            updated += 1

        except Exception as e:
            log.warning("  %-15s conversion failed: %s", row["site"], e)

    log.info("Updated %d nodes.", updated)
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    _root = os.environ.get("PYPSA_NZA_ROOT", "")
    if not _root:
        log.error("PYPSA_NZA_ROOT not set. See SETUP.md.")
        sys.exit(1)

    nodes_path = (Path(_root) / "pypsa_nza_workspace"
                  / "data" / "manual" / "nodes.csv")

    if not nodes_path.exists():
        log.error("nodes.csv not found: %s", nodes_path)
        sys.exit(1)

    # Backup
    backup_path = nodes_path.with_name("nodes_backup.csv")
    shutil.copy2(nodes_path, backup_path)
    log.info("Backup written: %s", backup_path)

    # Load -- strip unnamed columns caused by trailing commas in CSV
    df = pd.read_csv(nodes_path, dtype=str)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    for col in ("X", "Y", "lat", "long"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Initialise projection
    tm = TransverseMercator()
    init_tm_proj(tm)

    # Convert
    df = convert_nztm_to_wgs84(df, tm)

    # Save
    df.to_csv(nodes_path, index=False)
    log.info("Saved: %s", nodes_path)


if __name__ == "__main__":
    main()
