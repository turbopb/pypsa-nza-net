# -*- coding: utf-8 -*-
"""
Created on Sun May 31 15:54:02 2026

@author: OEM
"""
"""
nza_net_survey.py

Targeted survey of the pypsa-nza-net workspace.
Run from anywhere -- ROOT is hardcoded to the known workspace path.
Paste the full output back to Claude.

Usage (Spyder or PowerShell):
    python nza_net_survey.py
    %run nza_net_survey.py
"""

import sys
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# ROOT -- adjust if the workspace has moved
# ---------------------------------------------------------------------------

ROOT = Path(
    r"C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace"
)

NETWORKS_ROOT = ROOT / "models" / "networks" / "reference"

PREVIEW_ROWS = 3
MAX_COLS_DISPLAY = 120  # truncate long column lists in output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(char="=", w=72):
    print(char * w)


def preview_csv(path: Path, label: str = None, rows: int = PREVIEW_ROWS):
    label = label or path.name
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        n_data = max(0, len(lines) - 1)
        cols = lines[0][:MAX_COLS_DISPLAY] if lines else "?"
        print(f"      {label}  [{n_data} rows]")
        print(f"        cols : {cols}")
        for row in lines[1:rows + 1]:
            print(f"        data : {row[:MAX_COLS_DISPLAY]}")
    except Exception as e:
        print(f"      {label}  ERROR: {e}")


def count_rows(path: Path) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return max(0, len(lines) - 1)
    except Exception:
        return -1


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def classify_network(folder: Path) -> str:
    """
    Attempt to classify a network folder as combined / north / south / unknown
    by inspecting buses.csv bus names against known island-specific buses.
    """
    buses_file = folder / "buses.csv"
    if not buses_file.exists():
        return "unknown"
    try:
        text = buses_file.read_text(encoding="utf-8")
        names = {row.split(",")[0].strip() for row in text.splitlines()[1:] if row.strip()}
        # BEN is South Island (Benmore); HAY is North Island (Haywards)
        # Combined network must have both
        has_south = any(b in names for b in ["BEN", "INV", "ROX", "TWI", "ISL"])
        has_north = any(b in names for b in ["HAY", "HLY", "OTA", "HEN", "WGN"])
        n_buses = len(names)
        if has_south and has_north:
            return f"COMBINED ({n_buses} buses)"
        elif has_south:
            return f"south-only ({n_buses} buses)"
        elif has_north:
            return f"north-only ({n_buses} buses)"
        else:
            return f"unknown ({n_buses} buses)"
    except Exception as e:
        return f"error ({e})"


def is_pypsa_folder(folder: Path) -> bool:
    return (folder / "buses.csv").exists() and (folder / "lines.csv").exists()


def survey_network_folder(folder: Path, rel: Path):
    """Print a detailed summary of one PyPSA CSV network folder."""
    island_class = classify_network(folder)
    files = sorted(f.name for f in folder.iterdir() if f.is_file())

    print()
    print(f"  [{island_class}]  {rel}/")
    print(f"    files : {', '.join(files)}")

    # meta.json if present
    meta = read_json(folder / "meta.json") if (folder / "meta.json").exists() else {}
    if meta:
        print(f"    meta  : {meta}")

    # network.csv -- reference frequency/MVA base
    if (folder / "network.csv").exists():
        preview_csv(folder / "network.csv", "network.csv", rows=1)

    # snapshots
    snap = folder / "snapshots.csv"
    if snap.exists():
        n = count_rows(snap)
        try:
            lines = snap.read_text(encoding="utf-8").splitlines()
            first = lines[1].split(",")[0] if len(lines) > 1 else "?"
            last  = lines[-1].split(",")[0] if len(lines) > 1 else "?"
            print(f"      snapshots.csv  [{n} rows]  {first} .. {last}")
        except Exception:
            print(f"      snapshots.csv  [{n} rows]")

    # key static CSVs
    for fname in ["buses.csv", "lines.csv", "links.csv",
                  "generators.csv", "loads.csv", "transformers.csv"]:
        fpath = folder / fname
        if fpath.exists():
            preview_csv(fpath, fname)

    # time-varying CSVs
    for fname in ["loads-p_set.csv", "generators-p_max_pu.csv",
                  "generators-p_set.csv"]:
        fpath = folder / fname
        if fpath.exists():
            preview_csv(fpath, fname, rows=1)


# ---------------------------------------------------------------------------
# Section 1 -- workspace root overview
# ---------------------------------------------------------------------------

def survey_workspace_root():
    sep()
    print("SECTION 1 -- WORKSPACE ROOT")
    sep()
    print(f"ROOT         : {ROOT}")
    print(f"NETWORKS_ROOT: {NETWORKS_ROOT}")
    print(f"Python       : {sys.version}")
    print()

    if not ROOT.exists():
        print("ERROR: ROOT does not exist. Check the path at the top of this script.")
        return

    # Top-level directories only
    print("Top-level directories in ROOT:")
    for p in sorted(ROOT.iterdir()):
        marker = "[DIR] " if p.is_dir() else "[FILE]"
        print(f"  {marker} {p.name}")


# ---------------------------------------------------------------------------
# Section 2 -- networks/reference tree
# ---------------------------------------------------------------------------

def survey_networks_root():
    sep("-")
    print("SECTION 2 -- models/networks/reference TREE")
    sep("-")

    if not NETWORKS_ROOT.exists():
        print(f"ERROR: {NETWORKS_ROOT} does not exist.")
        print("  Check the networks path in this script.")
        return

    # Walk year-level directories
    for year_dir in sorted(NETWORKS_ROOT.iterdir()):
        if not year_dir.is_dir():
            continue
        print(f"\n  YEAR: {year_dir.name}/")

        # Walk variant-level directories under each year
        for variant_dir in sorted(year_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            rel = variant_dir.relative_to(NETWORKS_ROOT)

            if is_pypsa_folder(variant_dir):
                survey_network_folder(variant_dir, rel)
            else:
                # Recurse one more level (monthly subfolders?)
                sub_folders = [d for d in sorted(variant_dir.iterdir())
                               if d.is_dir() and is_pypsa_folder(d)]
                if sub_folders:
                    print(f"\n    {variant_dir.name}/  [{len(sub_folders)} sub-folders]")
                    for sf in sub_folders:
                        rel_sf = sf.relative_to(NETWORKS_ROOT)
                        survey_network_folder(sf, rel_sf)
                else:
                    print(f"\n    {variant_dir.name}/  (no PyPSA CSV folders found inside)")


# ---------------------------------------------------------------------------
# Section 3 -- dispatch config
# ---------------------------------------------------------------------------

def survey_dispatch_config():
    sep("-")
    print("SECTION 3 -- pypsa-nza-dispatch config")
    sep("-")

    config_path = Path(
        r"C:\Users\Public\Documents\Thesis\analysis\pypsa-nza-dispatch"
        r"\config\dispatch_config.yaml"
    )

    if not config_path.exists():
        print(f"  Not found: {config_path}")
        return

    print(f"  {config_path}")
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            print(f"    {line}")
    except Exception as e:
        print(f"  ERROR reading config: {e}")


# ---------------------------------------------------------------------------
# Section 4 -- cases/reference tree (heuristic working directories)
# ---------------------------------------------------------------------------

def survey_cases():
    sep("-")
    print("SECTION 4 -- cases/reference TREE (heuristic working directories)")
    sep("-")

    # This is where 01_setup_year.py writes its output
    # Derive from dispatch config path structure
    dispatch_root = Path(
        r"C:\Users\Public\Documents\Thesis\analysis\pypsa-nza-dispatch"
    )
    cases_root = dispatch_root / "cases" / "reference"

    print(f"  Expected cases root: {cases_root}")

    if not cases_root.exists():
        print("  Does not exist yet -- will be created by 01_setup_year.py")
        return

    for year_dir in sorted(cases_root.iterdir()):
        if not year_dir.is_dir():
            continue
        month_dirs = sorted(d for d in year_dir.iterdir() if d.is_dir())
        print(f"\n  {year_dir.name}/  [{len(month_dirs)} month directories]")
        for md in month_dirs:
            files = [f.name for f in md.iterdir() if f.is_file()]
            has_loads = "loads-p_set.csv" in files
            n_snap = count_rows(md / "loads-p_set.csv") if has_loads else 0
            island = classify_network(md)
            print(f"    {md.name}/  [{island}]  "
                  f"{'loads-p_set: ' + str(n_snap) + ' rows' if has_loads else 'NO loads-p_set.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sep()
    print("nza_net_survey.py -- pypsa-nza-net workspace survey")
    sep()

    survey_workspace_root()
    survey_networks_root()
    survey_dispatch_config()
    survey_cases()

    sep()
    print("Survey complete. Paste the full output back to Claude.")
    sep()