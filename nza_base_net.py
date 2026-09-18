#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nza_base_net.py

Build PyPSA network topology and load timeseries for NZ capacity expansion.

This script is part of the pypsa-nza-net module.  It reads clean, pre-processed
input data produced by pypsa-nza-data and constructs a PyPSA network with:

    - Buses from nodes.csv (110 kV and above AC; dedicated DC buses)
    - AC transmission lines from lines_data.csv (110 kV and 220 kV only),
      including optional explicit base-year ratings in ``s_nom_mva``
    - Transformers at multi-voltage substations (auto-detected)
    - One simplified Cook Strait HVDC connection (as a PyPSA Link)
    - Generators from generator_registry.csv (all dormant at p_max_pu=0)
    - Load timeseries from {year}_export_sites_all.csv

Island construction
-------------------
North Island, South Island, or full country networks can be built
independently.  The HVDC link is only added when building the full country
network (island=None).

Timestep and unit handling
--------------------------
The annual Electricity Authority/Transpower files must declare whether their
values are interval energy (MWh) or average power (MW).  Interval energy is
converted to average MW before it is assigned to PyPSA ``p_set``.  Snapshot
objective, generator and store weightings are the duration of one model
snapshot in hours.  This keeps both power balance and energy accounting
dimensionally correct for the native 30-minute data and for coarser models.

Network output
--------------
Networks are saved as PyPSA CSV folders under:
    {root}/models/networks/{scenario}/{year}/

Author: Phillippe Bruneau
"""

import sys
import logging
import calendar
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pypsa
import yaml


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

# Site codes that are earth return electrodes or ground references.
# These appear in nodes.csv but are not electrical network buses.
ELECTRODE_SITES = {"BGR", "THW"}


def get_dc_node_sites(df_links: pd.DataFrame) -> set:
    """
    Return the set of DC infrastructure bus site codes from links_data.csv.

    With the new single-link HVDC representation (BEN->HAY_220, both AC buses),
    there are no DC intermediate buses. This function returns an empty set.
    The old three-segment chain (BEN-DC, FTB, OTB, HAY-DC) is no longer used.

    Parameters
    ----------
    df_links : pd.DataFrame

    Returns
    -------
    set of str  (always empty with single-link HVDC)
    """
    return set()


# Minimum AC voltage (kV) for inclusion in the transmission network.
# Buses below this threshold can still carry load and generation but
# will not be endpoints of AC transmission lines.
AC_VOLT_THRESHOLD_KV = 110

# Transformer type for fixed 220/110 kV inter-bus transformers.
TRANSFORMER_TYPE = "630MVA 220/110kV"

# Custom transformer types not in the PyPSA standard library.
# Parameters based on 100 MVA 220/110 kV type, scaled to 630 MVA.
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


def register_custom_transformer_types(network: pypsa.Network) -> None:
    """Add custom transformer types to the network type library."""
    for name, params in CUSTOM_TRANSFORMER_TYPES.items():
        if name not in network.transformer_types.index:
            network.transformer_types.loc[name] = params


# Months list used for naming monthly network folders.
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

# Carrier definitions: name -> hex colour for PyPSA plotting.
CARRIERS = {
    "hydro":      "blue",
    "pumped_hydro": "royalblue",
    "solar_pv":   "yellow",
    "geothermal": "orange",
    "coal":       "black",
    "coal_gas":   "dimgray",
    "gas":        "green",
    "diesel":     "brown",
    "wind":       "cyan",
    "off_wind":   "dodgerblue",
    "ocgt":       "purple",
    "ccgt":       "violet",
    "hydrogen":   "pink",
    "battery":    "gold",
    "bioenergy":  "lime",
    "AC":         "gray",
    "DC":         "red",
}


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: Path) -> Dict:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : Path

    Returns
    -------
    dict
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fixed_interval_hours(frequency: str) -> float:
    """Return the duration in hours of a fixed pandas frequency."""
    try:
        offset = pd.tseries.frequencies.to_offset(frequency)
        hours = offset.nanos / 3_600_000_000_000
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Frequency {frequency!r} is not a fixed-duration interval"
        ) from exc
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError(f"Frequency {frequency!r} has invalid duration")
    return float(hours)


def annual_timeseries_contract(config: Dict) -> Dict[str, str]:
    """Validate the explicit source quantity/unit/frequency declaration."""
    contract = config.get("annual_timeseries")
    if not isinstance(contract, dict):
        raise ValueError(
            "Config must define annual_timeseries with input_quantity, "
            "input_unit and source_interval"
        )

    quantity = str(contract.get("input_quantity", "")).strip().lower()
    unit = str(contract.get("input_unit", "")).strip()
    source_interval = str(contract.get("source_interval", "")).strip()
    valid_pairs = {
        ("interval_energy", "MWh"),
        ("average_power", "MW"),
    }
    if (quantity, unit) not in valid_pairs:
        raise ValueError(
            "annual_timeseries must declare either "
            "input_quantity=interval_energy with input_unit=MWh, or "
            "input_quantity=average_power with input_unit=MW"
        )
    if not source_interval:
        raise ValueError("annual_timeseries.source_interval is required")
    fixed_interval_hours(source_interval)
    return {
        "input_quantity": quantity,
        "input_unit": unit,
        "source_interval": source_interval,
    }


def resolve_paths(config: Dict) -> Dict:
    """
    Build absolute Path objects for all data inputs.

    The config 'paths.root' key is the workspace root.  All other path
    keys are relative to root.  Returns a flat dict of Path objects.

    Parameters
    ----------
    config : dict

    Returns
    -------
    dict of {str: Path}
    """
    root = Path(config["paths"]["root"])

    return {
        "nodes":    root / config["paths"]["dirpath_manual"]  / config["filename"]["nodes"],
        "lines_si": root / config["paths"]["dirpath_manual"]  / config["filename"].get("lines_si", "lines_data_si.csv"),
        "lines_ni": root / config["paths"]["dirpath_manual"]  / config["filename"].get("lines_ni", "lines_data_ni.csv"),
        "lines":    root / config["paths"]["dirpath_manual"]  / config["filename"]["lines"],
        "links":    root / config["paths"]["dirpath_manual"]  / config["filename"]["links"],
        "gen_reg":  root / config["paths"]["dirpath_static"]  / config["filename"]["generator_registry"],
        "annual":   root / config["paths"]["dirpath_annual"],
        "networks": root / config["paths"]["dirpath_networks"],
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_nodes(nodes_path: Path) -> pd.DataFrame:
    """
    Load and validate the nodes (bus) definition file.

    Filters out electrode stations.  Retains all voltage levels -- the
    caller decides which buses to include based on context.

    Parameters
    ----------
    nodes_path : Path

    Returns
    -------
    pd.DataFrame
        Columns: name, site, zone, island, volts, X, Y, long, lat
    """
    df = pd.read_csv(nodes_path, dtype=str)

    # Strip leading/trailing whitespace from string columns
    for col in df.columns:
        df[col] = df[col].str.strip()

    # Convert numeric columns
    df["volts"] = pd.to_numeric(df["volts"], errors="coerce")
    df["long"]  = pd.to_numeric(df["long"],  errors="coerce")
    df["lat"]   = pd.to_numeric(df["lat"],   errors="coerce")

    # Drop electrode stations
    n_before = len(df)
    df = df[~df["site"].isin(ELECTRODE_SITES)].copy()
    if len(df) < n_before:
        log.info("  Excluded %d electrode stations", n_before - len(df))

    df = df.reset_index(drop=True)
    log.info("  Loaded %d bus definitions", len(df))
    return df


def load_lines(lines_path: Path) -> pd.DataFrame:
    """
    Load transmission line definitions.

    Filters to AC lines only (volts < 350 kV) at or above the AC voltage
    threshold.  The 'distance' column is used as line length (circuit km).

    Parameters
    ----------
    lines_path : Path

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(lines_path, dtype=str)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    df["volts"] = pd.to_numeric(df["volts"], errors="coerce")

    # Determine circuit length column using fallback logic:
    #   1. Use 'distance' if the column exists and is complete (no NaN)
    #   2. Fall back to 'length' if 'distance' is absent or incomplete
    #   3. Abort if 'length' is also absent or has NaN values
    has_distance = "distance" in df.columns
    has_length   = "length"   in df.columns

    if has_distance:
        df["_dist_raw"] = pd.to_numeric(df["distance"], errors="coerce")
        n_dist_nan = df["_dist_raw"].isna().sum()
    else:
        n_dist_nan = len(df)

    if has_length:
        df["_len_raw"] = pd.to_numeric(df["length"], errors="coerce")
        n_len_nan = df["_len_raw"].isna().sum()
    else:
        n_len_nan = len(df)

    if has_distance and n_dist_nan == 0:
        df["distance"] = df["_dist_raw"]
        log.info("  Circuit length: using 'distance' column (QGIS-measured, complete)")
    elif has_length and n_len_nan == 0:
        df["distance"] = df["_len_raw"]
        if has_distance and n_dist_nan > 0:
            missing = df[df["_dist_raw"].isna()]["line"].tolist()
            log.warning("  'distance' column incomplete (%d missing): %s",
                        n_dist_nan, missing)
        log.info("  Circuit length: using 'length' column (straight-line estimates)")
        log.info("  NOTE: straight-line lengths are a lower bound on true circuit")
        log.info("        length.  Impedances will be underestimated accordingly.")
    else:
        missing_len = df[df["_len_raw"].isna()]["line"].tolist() if has_length else "all"
        raise ValueError(
            f"No complete circuit length column found in lines_data.csv.\n"
            f"  'distance': {'incomplete' if has_distance else 'absent'}\n"
            f"  'length':   {'incomplete (missing: ' + str(missing_len) + ')' if has_length else 'absent'}\n"
            f"Ensure at least one column is complete before running."
        )

    # Drop working columns
    for col in ("_dist_raw", "_len_raw"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # AC lines only: below 350 kV (HVDC threshold) and at/above 110 kV
    n_before = len(df)
    df = df[
        df["volts"].notna() &
        (df["volts"] < 350) &
        (df["volts"] >= AC_VOLT_THRESHOLD_KV)
    ].copy()

    log.info("  Loaded %d AC transmission lines (excluded %d below threshold or DC)",
             len(df), n_before - len(df))
    return df


def load_links(links_path: Path) -> pd.DataFrame:
    """
    Load HVDC link definitions.

    Parameters
    ----------
    links_path : Path

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(links_path, dtype=str)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    df["length"]     = pd.to_numeric(df["length"],     errors="coerce")
    df["build_year"] = pd.to_numeric(df["build_year"], errors="coerce")
    if "efficiency" in df.columns:
        df["efficiency"] = pd.to_numeric(df["efficiency"], errors="coerce")

    log.info("  Loaded %d HVDC links", len(df))
    return df


def load_generator_registry(gen_path: Path) -> pd.DataFrame:
    """
    Load the processed generator registry.

    Parameters
    ----------
    gen_path : Path

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(gen_path, dtype=str)
    df["p_nom_mw"]   = pd.to_numeric(df["p_nom_mw"],   errors="coerce")
    df["volts_kv"]   = pd.to_numeric(df["volts_kv"],   errors="coerce")
    df["build_year"] = pd.to_numeric(df["build_year"], errors="coerce").fillna(0).astype(int)
    log.info("  Loaded %d generator entries", len(df))
    return df


def load_annual_timeseries(
    annual_dir: Path,
    year: int,
    freq_source: str = "30min",
    freq_target: str = "30min",
    input_quantity: str = "interval_energy",
    input_unit: str = "MWh",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load annual export (load) and import (generation) timeseries.

    Resamples from source frequency to target frequency if they differ, then
    returns average power in MW. For interval-energy input, source MWh are
    summed into the target interval and divided by target hours. For
    average-power input, MW values are averaged into the target interval.

    Parameters
    ----------
    annual_dir : Path
        Directory containing {year}_export_sites_all.csv etc.
    year : int
    freq_source : str
        Native data frequency (default '30min').
    freq_target : str
        Target frequency for the network (e.g. '30min', '1h', '1D').
    input_quantity : str
        ``interval_energy`` or ``average_power``.
    input_unit : str
        ``MWh`` for interval energy or ``MW`` for average power.

    Returns
    -------
    (df_export, df_import) : tuple of pd.DataFrame
        Both indexed by datetime, columns are site codes, values are MW.
    """
    export_path = annual_dir / f"{year}_export_sites_all.csv"
    import_path = annual_dir / f"{year}_import_sites_all.csv"

    for p in (export_path, import_path):
        if not p.exists():
            raise FileNotFoundError(f"Timeseries file not found: {p}")

    df_export = pd.read_csv(export_path, index_col=0, parse_dates=True)
    df_import = pd.read_csv(import_path, index_col=0, parse_dates=True)

    quantity = str(input_quantity).strip().lower()
    unit = str(input_unit).strip()
    valid_pairs = {
        ("interval_energy", "MWh"),
        ("average_power", "MW"),
    }
    if (quantity, unit) not in valid_pairs:
        raise ValueError(
            f"Unsupported annual timeseries declaration: {quantity!r}, {unit!r}"
        )

    source_hours = fixed_interval_hours(freq_source)
    target_hours = fixed_interval_hours(freq_target)
    if target_hours < source_hours:
        raise ValueError(
            f"Upsampling annual data is not supported: {freq_source} -> {freq_target}"
        )
    ratio = target_hours / source_hours
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Target interval {freq_target} is not an integer multiple of "
            f"source interval {freq_source}"
        )

    for label, dataframe in (("Export", df_export), ("Import", df_import)):
        dataframe.index = pd.DatetimeIndex(
            pd.to_datetime(dataframe.index, errors="raise")
        )
        if dataframe.index.has_duplicates:
            raise ValueError(f"{label} timeseries contains duplicate timestamps")
        if not dataframe.index.is_monotonic_increasing:
            raise ValueError(f"{label} timeseries timestamps are not ordered")
        dataframe.columns = [str(column).strip().upper() for column in dataframe.columns]
        if pd.Index(dataframe.columns).duplicated().any():
            raise ValueError(f"{label} timeseries contains duplicate site columns")
        converted = dataframe.apply(pd.to_numeric, errors="raise")
        if converted.isna().any().any():
            raise ValueError(f"{label} timeseries contains missing values")
        if (converted < 0).any().any():
            raise ValueError(f"{label} timeseries contains negative values")
        if label == "Export":
            df_export = converted
        else:
            df_import = converted

    log.info("  Export: %d timesteps x %d sites", len(df_export), len(df_export.columns))
    log.info("  Import: %d timesteps x %d sites", len(df_import), len(df_import.columns))

    if quantity == "interval_energy":
        if freq_target != freq_source:
            log.info("  Resampling %s -> %s (sum for interval energy)", freq_source, freq_target)
            df_export = df_export.resample(freq_target).sum()
            df_import = df_import.resample(freq_target).sum()
        df_export = df_export / target_hours
        df_import = df_import / target_hours
        log.info(
            "  Converted interval energy (%s) to average power (MW) using %.6g h",
            unit,
            target_hours,
        )
    elif freq_target != freq_source:
        log.info("  Resampling %s -> %s (mean for average power)", freq_source, freq_target)
        df_export = df_export.resample(freq_target).mean()
        df_import = df_import.resample(freq_target).mean()

    if freq_target != freq_source:
        log.info("  After resample: %d timesteps", len(df_export))

    return df_export, df_import


# =============================================================================
# MULTI-VOLTAGE SITE DETECTION
# =============================================================================

def find_multi_voltage_sites(df_lines: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Identify sites that appear as endpoints on lines at more than one voltage.

    These sites require separate buses per voltage level connected by a
    transformer in the PyPSA model.

    Parameters
    ----------
    df_lines : pd.DataFrame
        Must have columns 'bus0', 'bus1', 'volts'.

    Returns
    -------
    dict of {site_code: [voltage_list]}
        Only sites with more than one voltage are returned.
    """
    site_volts: Dict[str, set] = {}

    for _, row in df_lines.iterrows():
        v = int(row["volts"])
        for bus_col in ("bus0", "bus1"):
            site = row[bus_col]
            if site not in site_volts:
                site_volts[site] = set()
            site_volts[site].add(v)

    multi = {
        site: sorted(volts)
        for site, volts in site_volts.items()
        if len(volts) > 1
    }

    if multi:
        log.info("  Multi-voltage sites (%d): %s", len(multi), sorted(multi.keys()))

    return multi


def resolve_bus_name(site: str, volts: Optional[int],
                     multi_volt_sites: Dict) -> str:
    """
    Return the PyPSA bus name for a given site and voltage.

    Single-voltage sites use the plain site code.
    Multi-voltage sites use SITE_VVV notation.

    Parameters
    ----------
    site : str
    volts : int or None
    multi_volt_sites : dict

    Returns
    -------
    str
    """
    if site in multi_volt_sites and volts is not None:
        return f"{site}_{int(volts)}"
    return site


# =============================================================================
# NETWORK COMPONENT BUILDERS
# =============================================================================

def add_carriers(network: pypsa.Network) -> pypsa.Network:
    """Add all technology carriers with associated colours."""
    for carrier, color in CARRIERS.items():
        if carrier not in network.carriers.index:
            network.add("Carrier", carrier, color=color, co2_emissions=0.0)
    return network


def add_buses(
    network: pypsa.Network,
    df_nodes: pd.DataFrame,
    df_lines: pd.DataFrame,
    multi_volt_sites: Dict,
    dc_sites: set,
    island: Optional[str] = None,
) -> pypsa.Network:
    """
    Add buses to the network from nodes.csv.

    For single-voltage sites, one bus is created with the site code as name.
    For multi-voltage sites (detected from lines_data.csv), one bus per
    voltage level is created with suffixed names (e.g. ROX_110, ROX_220).

    The HVDC DC buses (BEN-DC, HAY-DC) are added at carrier='DC'.
    All AC buses are added at carrier='AC'.

    Parameters
    ----------
    network : pypsa.Network
    df_nodes : pd.DataFrame
    multi_volt_sites : dict
        From find_multi_voltage_sites().
    island : str or None
        'NI', 'SI', or None (all).

    Returns
    -------
    pypsa.Network
    """
    log.info("\n  Adding buses")

    # Filter by island
    if island is not None:
        df = df_nodes[df_nodes["island"] == island].copy()
        log.info("    Island filter '%s': %d/%d buses", island, len(df), len(df_nodes))
    else:
        df = df_nodes.copy()

    # Further filter to only sites that appear as line or link endpoints.
    # nodes.csv is a master register -- only sites referenced in
    # lines_data_si.csv (or links_data.csv) are actual network buses.
    # Junction nodes (JCT_*) and DC sites are included automatically.
    transmission_sites = set()
    for _, row in df_lines.iterrows():
        for col in ("bus0", "bus1"):
            s = str(row[col]).strip().upper()
            transmission_sites.add(s)
            # Also add the base site without voltage suffix
            base = s.rsplit("_", 1)[0] if "_" in s else s
            transmission_sites.add(base)
    for site in dc_sites:
        transmission_sites.add(site)

    df = df[df["site"].str.upper().isin(transmission_sites)].copy()
    log.info("    Line endpoint filter: %d transmission buses", len(df))

    # Build voltage lookup from lines data -- authoritative for transmission voltage.
    # nodes.csv volts column reflects distribution connection voltage which
    # is not the same as the transmission bus voltage. For single-voltage
    # sites, use the voltage from the line endpoints. For multi-voltage sites,
    # the voltage is already handled by multi_volt_sites.
    #
    # JCT_ junction nodes must NOT be base-stripped -- their full name is
    # their site code. Only strip the voltage suffix for multi-voltage sites
    # like ROX_220 or SFD_110 (numeric suffix, 2-3 digits).
    import re as _re
    _volt_suffix = _re.compile(r"_(\d{2,3})$")

    site_volt_from_lines: Dict[str, int] = {}
    for _, lrow in df_lines.iterrows():
        v = int(lrow["volts"])
        for col in ("bus0", "bus1"):
            s = str(lrow[col]).strip()
            # Strip numeric voltage suffix (e.g. ROX_220 -> ROX)
            # but keep JCT_ names and plain site codes intact
            m = _volt_suffix.search(s)
            base = s[:m.start()] if m else s
            if base not in site_volt_from_lines:
                site_volt_from_lines[base] = v
            # If site appears at multiple voltages it will be in
            # multi_volt_sites and handled separately -- keep highest voltage
            # to avoid assigning sub-transmission voltage to a 220 kV bus
            else:
                site_volt_from_lines[base] = max(site_volt_from_lines[base], v)

    added = 0
    skipped = 0

    for _, row in df.iterrows():
        site    = row["site"]
        long    = row["long"]
        lat     = row["lat"]

        # DC buses: any site appearing in links_data.csv as a DC endpoint
        # is forced to carrier='DC' and v_nom=350, overriding nodes.csv.
        # This handles intermediate nodes (FTB, OTB) that have no voltage
        # entry in nodes.csv alongside named converter stations (BEN-DC, HAY-DC).
        if site in dc_sites:
            bus_name = site
            if bus_name not in network.buses.index:
                network.add("Bus", bus_name,
                            v_nom=350.0, carrier="DC",
                            x=long, y=lat)
                added += 1
            continue

        # AC buses -- voltage from lines data, not nodes.csv
        carrier = "AC"
        v_nom = float(site_volt_from_lines.get(site, 0))

        if site in multi_volt_sites:
            # Create one bus per voltage level for this site
            for v in multi_volt_sites[site]:
                bus_name = f"{site}_{v}"
                if bus_name not in network.buses.index:
                    network.add("Bus", bus_name,
                                v_nom=float(v), carrier=carrier,
                                x=long, y=lat)
                    added += 1
        else:
            bus_name = site
            if bus_name not in network.buses.index:
                network.add("Bus", bus_name,
                            v_nom=v_nom, carrier=carrier,
                            x=long, y=lat)
                added += 1

    log.info("    Added %d buses (%d skipped as duplicates)", added, skipped)
    return network


def add_lines(
    network: pypsa.Network,
    df_lines: pd.DataFrame,
    multi_volt_sites: Dict,
    config: Optional[Dict] = None,
) -> pypsa.Network:
    """
    Add AC transmission lines to the network.

    Line electrical parameters (r, x, b) are computed by PyPSA from the
    standard line type library and the circuit length.  The 'distance'
    column from lines_data.csv is used as the circuit length in km.

    Double-circuit lines (config == 'double') are represented with
    num_parallel=2. If a source row contains a non-blank ``s_nom_mva`` value,
    that explicit source-data rating is used instead of the generic rating
    inferred from the standard line type. These are base-network data
    corrections for assets already represented in the reference network;
    they are not future transmission investments.

    Parameters
    ----------
    network : pypsa.Network
    df_lines : pd.DataFrame
    multi_volt_sites : dict
    config : dict, optional
        Retained for call-site compatibility. Line ratings are read from the
        authoritative transmission source table rather than configuration.

    Returns
    -------
    pypsa.Network
    """
    log.info("\n  Adding AC transmission lines")

    # Map voltage (kV) to PyPSA standard line type
    LINE_TYPE_MAP = {
        220: "490-AL1/64-ST1A 220.0",
        110: "305-AL1/39-ST1A 110.0",
    }
    default_type = "490-AL1/64-ST1A 220.0"

    added   = 0
    skipped = 0

    for _, row in df_lines.iterrows():
        line_name = row["line"]
        volts     = int(row["volts"])
        bus0_site = row["bus0"]
        bus1_site = row["bus1"]
        length_km = row["distance"]   # circuit km, not straight-line

        bus0 = resolve_bus_name(bus0_site, volts, multi_volt_sites)
        bus1 = resolve_bus_name(bus1_site, volts, multi_volt_sites)

        # Skip if either bus is not in network
        if bus0 not in network.buses.index:
            log.warning("    Line %s: bus0 '%s' not in network -- skipped",
                        line_name, bus0)
            skipped += 1
            continue
        if bus1 not in network.buses.index:
            log.warning("    Line %s: bus1 '%s' not in network -- skipped",
                        line_name, bus1)
            skipped += 1
            continue

        line_type   = LINE_TYPE_MAP.get(volts, default_type)
        num_parallel = 2 if str(row.get("config", "")).lower() == "double" else 1

        # A PyPSA line type supplies impedance and current-rating data, but it
        # does not set the Line component's optimisation capacity.  Fixed
        # lines therefore require an explicit s_nom; otherwise the Line
        # default of zero MVA prevents all power transfer in n.optimize().
        if line_type not in network.line_types.index:
            raise KeyError(
                f"Line {line_name}: unknown PyPSA line type {line_type!r}"
            )

        i_nom_ka = float(network.line_types.at[line_type, "i_nom"])
        if not math.isfinite(i_nom_ka) or i_nom_ka <= 0.0:
            raise ValueError(
                f"Line {line_name}: line type {line_type!r} has invalid "
                f"i_nom={i_nom_ka!r} kA"
            )

        # kV * kA gives MVA for the balanced three-phase rating.
        generic_s_nom_mva = (
            math.sqrt(3.0)
            * float(volts)
            * i_nom_ka
            * float(num_parallel)
        )
        explicit_rating_raw = row.get("s_nom_mva", "")
        explicit_rating_text = (
            "" if pd.isna(explicit_rating_raw)
            else str(explicit_rating_raw).strip()
        )
        if explicit_rating_text:
            try:
                s_nom_mva = float(explicit_rating_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Line {line_name}: s_nom_mva {explicit_rating_raw!r} "
                    "is not numeric"
                ) from exc
            if not math.isfinite(s_nom_mva) or s_nom_mva <= 0.0:
                raise ValueError(
                    f"Line {line_name}: s_nom_mva must be finite and positive, "
                    f"got {s_nom_mva!r}"
                )
            log.info(
                "    Line %s: explicit source-data rating "
                "%.6f MVA (generic proxy %.6f MVA)",
                line_name,
                s_nom_mva,
                generic_s_nom_mva,
            )
        else:
            s_nom_mva = generic_s_nom_mva

        network.add(
            "Line",
            line_name,
            bus0=bus0,
            bus1=bus1,
            length=length_km,
            type=line_type,
            num_parallel=num_parallel,
            carrier="AC",
            s_nom=s_nom_mva,
            s_nom_extendable=False,
        )
        added += 1

    log.info("    Added %d lines (%d skipped)", added, skipped)
    return network


def add_transformers(
    network: pypsa.Network,
    multi_volt_sites: Dict,
    df_nodes: pd.DataFrame,
) -> pypsa.Network:
    """
    Add transformers at multi-voltage substations.

    For each site in multi_volt_sites, a transformer is added between the
    highest and lowest voltage buses at that site.  The PyPSA standard
    transformer type is used.  s_nom is extendable with capital_cost=0
    as placeholder.

    Parameters
    ----------
    network : pypsa.Network
    multi_volt_sites : dict
        {site: [v_low, v_high]} from find_multi_voltage_sites().
    df_nodes : pd.DataFrame

    Returns
    -------
    pypsa.Network
    """
    register_custom_transformer_types(network)
    log.info("\n  Adding transformers at multi-voltage substations")

    # Build lookup of available transformer types from PyPSA
    available_types = list(network.transformer_types.index) if len(network.transformer_types) > 0 else []

    added = 0
    for site, volts in multi_volt_sites.items():
        v_low  = min(volts)
        v_high = max(volts)
        bus_low  = f"{site}_{v_low}"
        bus_high = f"{site}_{v_high}"

        if bus_low not in network.buses.index or bus_high not in network.buses.index:
            log.warning("    Transformer %s: one or both buses missing -- skipped", site)
            continue

        trafo_name = f"TRAFO_{site}_{v_high}_{v_low}"

        # Select best matching standard type or use specified type
        trafo_type = TRANSFORMER_TYPE

        network.add(
            "Transformer",
            trafo_name,
            bus0=bus_high,
            bus1=bus_low,
            type=trafo_type,
            s_nom_extendable=False,
            capital_cost=0.0,
        )
        log.info("    %s: %s kV <-> %s kV  [%s]", trafo_name, v_high, v_low, trafo_type)
        added += 1

    log.info("    Added %d transformers", added)
    return network


def add_hvdc_links(
    network: pypsa.Network,
    df_links: pd.DataFrame,
    island: Optional[str],
) -> pypsa.Network:
    """
    Add Cook Strait HVDC link (single BEN->HAY_220 Link).

    Replaces the previous three-segment chain (BEN-DC->FTB->OTB->HAY-DC)
    with a single bidirectional Link between Benmore AC (BEN, 220 kV) and
    Haywards AC (HAY_220, 220 kV). DC intermediate buses are not added.
    No carrier is set to avoid DC sub-network topology issues in the LP solver.
    """
    if island is not None:
        log.info("\n  HVDC links: skipped (single-island build)")
        return network

    log.info("\n  Adding HVDC Cook Strait link (single BEN->HAY_220)")

    HVDC_BUS0       = "BEN"
    HVDC_BUS1       = "HAY_220"
    HVDC_P_NOM      = 1200.0
    HVDC_EFFICIENCY = 0.965

    if HVDC_BUS0 not in network.buses.index:
        log.warning("  HVDC skipped: bus %s not in network", HVDC_BUS0)
        return network
    if HVDC_BUS1 not in network.buses.index:
        log.warning("  HVDC skipped: bus %s not in network", HVDC_BUS1)
        return network

    network.add(
        "Link",
        "HVDC_BEN_HAY",
        bus0=HVDC_BUS0,
        bus1=HVDC_BUS1,
        p_nom=HVDC_P_NOM,
        p_min_pu=-1.0,
        p_max_pu=1.0,
        efficiency=HVDC_EFFICIENCY,
    )

    log.info("  Added HVDC_BEN_HAY: %s -> %s  p_nom=%.0f MW  eta=%.4f",
             HVDC_BUS0, HVDC_BUS1, HVDC_P_NOM, HVDC_EFFICIENCY)
    return network


def prepare_model_generators(
    df_gen: pd.DataFrame,
    multi_volt_sites: Dict,
) -> pd.DataFrame:
    """Convert the global generator registry into the model inventory.

    Huntly is deliberately represented as one simplified gas generator. The
    registry's three HLY records are therefore aggregated by capacity rather
    than added as separate technology-specific units. Its mixed commissioning
    history is represented by build_year=0 so retirement remains a TEFF
    planner decision instead of being inferred from a false single vintage.
    """
    model_gen = df_gen.copy()
    model_gen["site"] = model_gen["site"].astype(str).str.strip().str.upper()

    hly_mask = model_gen["site"].eq("HLY")
    if not hly_mask.any():
        raise ValueError("Global generator registry contains no HLY entries")

    hly_rows = model_gen.loc[hly_mask]
    hly = hly_rows.iloc[0].copy()
    hly["plant_name"] = "Huntly Power Station (lumped)"
    hly["site"] = "HLY"
    hly["volts_kv"] = 220
    hly["bus"] = "HLY"
    hly["carrier"] = "gas"
    hly["p_nom_mw"] = float(hly_rows["p_nom_mw"].sum())
    hly["build_year"] = 0
    if "peaking" in hly.index:
        hly["peaking"] = "N"

    model_gen = pd.concat(
        [model_gen.loc[~hly_mask], pd.DataFrame([hly])],
        ignore_index=True,
    )

    def _resolve_generator_bus(row: pd.Series) -> str:
        site = str(row["site"]).strip()
        volts = None if pd.isna(row["volts_kv"]) else int(row["volts_kv"])
        return resolve_bus_name(site, volts, multi_volt_sites)

    model_gen["_resolved_bus"] = model_gen.apply(
        _resolve_generator_bus, axis=1
    )
    model_gen["_generator_name"] = (
        model_gen["_resolved_bus"].astype(str)
        + "_"
        + model_gen["carrier"].astype(str)
    )

    log.info(
        "  Generator model inventory: %d registry rows -> %d model generators; "
        "HLY lumped capacity %.1f MW",
        len(df_gen),
        len(model_gen),
        float(hly["p_nom_mw"]),
    )
    return model_gen


def add_generators(
    network: pypsa.Network,
    df_gen: pd.DataFrame,
    df_lines: pd.DataFrame,
    multi_volt_sites: Dict,
    island: Optional[str] = None,
) -> pypsa.Network:
    """
    Add generators to the network from the generator registry.

    All generators are added dormant (p_max_pu=0).  They are activated
    by nza_custom_net.py during the customisation step.

    Parameters
    ----------
    network : pypsa.Network
    df_gen : pd.DataFrame
        Global generator registry from build_generator_registry.py.
    df_lines : pd.DataFrame
        AC transmission lines used to construct this network.
    multi_volt_sites : dict

    Returns
    -------
    pypsa.Network
    """
    log.info("\n  Adding generators (all dormant)")

    model_gen = prepare_model_generators(df_gen, multi_volt_sites)

    # Resolve the source line endpoints exactly as add_lines() does. This
    # provides an explicit source-data connectivity check for each generator.
    connected_line_buses = set()
    for _, line in df_lines.iterrows():
        volts = int(line["volts"])
        for endpoint in ("bus0", "bus1"):
            connected_line_buses.add(
                resolve_bus_name(
                    str(line[endpoint]).strip(), volts, multi_volt_sites
                )
            )

    # Pre-filter to generators whose resolved bus exists in this network.
    # When building a single-island network, NI generators are silently
    # excluded rather than generating a warning per skipped generator.
    network_buses = set(network.buses.index)
    df_gen_filtered = model_gen[
        model_gen["_resolved_bus"].isin(network_buses)
    ].copy()
    n_excluded = len(model_gen) - len(df_gen_filtered)

    duplicate_names = df_gen_filtered.loc[
        df_gen_filtered["_generator_name"].duplicated(keep=False),
        "_generator_name",
    ].unique()
    if len(duplicate_names):
        raise ValueError(
            "Duplicate generator names in the selected network: "
            + ", ".join(str(name) for name in duplicate_names)
        )

    # A combined NZ network must contain every registered generator. Missing
    # entries indicate a real registry-to-topology propagation error.
    if island is None and n_excluded:
        missing = model_gen.loc[
            ~model_gen["_resolved_bus"].isin(network_buses),
            ["plant_name", "site", "volts_kv", "_resolved_bus"],
        ]
        details = "; ".join(
            f"{row['plant_name']} -> {row['_resolved_bus']}"
            for _, row in missing.iterrows()
        )
        raise ValueError(
            "Combined network is missing generator buses: " + details
        )

    unconnected = df_gen_filtered.loc[
        ~df_gen_filtered["_resolved_bus"].isin(connected_line_buses),
        ["plant_name", "_resolved_bus"],
    ]
    if not unconnected.empty:
        details = "; ".join(
            f"{row['plant_name']} -> {row['_resolved_bus']}"
            for _, row in unconnected.iterrows()
        )
        raise ValueError(
            "Generator buses are not connected in the source line data: "
            + details
        )

    if n_excluded > 0:
        log.info("    Excluded %d generators not in this island network", n_excluded)

    added   = 0
    skipped = 0

    for _, row in df_gen_filtered.iterrows():
        bus_name = row["_resolved_bus"]

        if bus_name not in network.buses.index:
            log.warning("    Generator at bus '%s' not in network -- skipped", bus_name)
            skipped += 1
            continue

        carrier    = row["carrier"]
        gen_name   = row["_generator_name"]
        p_nom      = float(row["p_nom_mw"])
        build_year = int(row["build_year"])

        network.add(
            "Generator",
            gen_name,
            bus=bus_name,
            p_nom=p_nom,
            carrier=carrier,
            build_year=build_year,
            efficiency=1.0,
            p_max_pu=0.0,   # dormant -- activated by nza_custom_net.py
            p_min_pu=0.0,
        )
        added += 1

    log.info("    Added %d generators (%d skipped: bus not in network)", added, skipped)
    return network


# =============================================================================
# LOAD TIMESERIES
# =============================================================================

def add_loads(
    network: pypsa.Network,
    df_export: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    df_nodes: pd.DataFrame,
    config: dict = None,
    island: str = None,
) -> pypsa.Network:
    """
    Add load timeseries to the network.

    ALL export sites are included as loads. Sites that are transmission
    buses receive their load directly. Sites that are not in the network
    (sub-transmission GXPs) are automatically aggregated to their nearest
    connected transmission bus by straight-line distance.

    This ensures the total modelled load equals total measured generation
    plus HVDC import, allowing a balanced power flow verification.
    """
    import math

    log.info("\n  Adding loads")

    # Build coordinate lookup for all nodes
    node_coords = {}
    for _, row in df_nodes.iterrows():
        site = str(row["site"]).strip().upper()
        try:
            lat  = float(row["lat"])
            lon  = float(row["long"])
            if not (math.isnan(lat) or math.isnan(lon)):
                node_coords[site] = (lat, lon)
        except (ValueError, TypeError):
            pass

    # Network buses (AC transmission only -- exclude DC buses)
    ac_net_buses = set(
        b for b in network.buses.index
        if network.buses.loc[b, "carrier"] == "AC"
    )

    # Build aggregation map: for each export site, find target bus
    # Manual overrides from config take precedence
    manual_agg = {}
    if config:
        for child, parent in config.get(
                "sub_transmission_aggregation", {}).items():
            manual_agg[child.upper()] = parent.upper()

    def nearest_bus(site: str) -> str:
        """Find nearest connected AC transmission bus by straight-line distance."""
        if site in node_coords:
            lat1, lon1 = node_coords[site]
            best_bus  = None
            best_dist = float("inf")
            for bus in ac_net_buses:
                # Strip voltage suffix for coord lookup
                base = bus.rsplit("_", 1)[0] if "_" in bus else bus
                coords = node_coords.get(bus) or node_coords.get(base)
                if coords is None:
                    continue
                lat2, lon2 = coords
                dist = math.sqrt((lat2-lat1)**2 + (lon2-lon1)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_bus  = bus
            return best_bus
        return None

    # Filter export timeseries to this island's sites only.
    # The all-sites export file contains both NI and SI GXPs.
    # Using the full file would assign NI loads to SI buses.
    if island is not None:
        island_sites = set(
            str(row["site"]).strip().upper()
            for _, row in df_nodes.iterrows()
            if str(row.get("island", "")).strip().upper() == island
        )
        export_cols = [c for c in df_export.columns
                       if c.strip().upper() in island_sites]
        df_export_island = df_export[export_cols]
        log.info("    Export filter '%s': %d/%d sites",
                 island, len(export_cols), len(df_export.columns))
    else:
        df_export_island = df_export

    # Align export data to snapshots
    df_slice = df_export_island.reindex(snapshots).fillna(0.0)

    # Accumulate monthly and full-year load per target bus. The annual series
    # distinguishes a genuinely inactive load location from an active location
    # that happens to have zero demand in the current month.
    bus_load = {}          # bus -> monthly pd.Series
    annual_bus_load = {}   # bus -> annual pd.Series
    agg_count   = 0
    direct_count = 0
    skip_count  = 0

    for site in df_slice.columns:
        site_up = site.strip().upper()

        # Determine target bus
        if site_up in manual_agg:
            target = manual_agg[site_up]
        elif site_up in ac_net_buses:
            target = site_up
        else:
            # Auto-aggregate to nearest transmission bus
            target = nearest_bus(site_up)
            if target is None:
                log.debug("    Load %s: no nearby bus -- skipped", site)
                skip_count += 1
                continue

        if target not in ac_net_buses:
            skip_count += 1
            continue

        annual_series = df_export_island[site].fillna(0.0)

        if target not in bus_load:
            bus_load[target] = df_slice[site].copy()
            annual_bus_load[target] = annual_series.copy()
        else:
            bus_load[target] = bus_load[target] + df_slice[site]
            annual_bus_load[target] = annual_bus_load[target] + annual_series

        if target == site_up:
            direct_count += 1
        else:
            agg_count += 1

    # Add aggregated loads to network. Exclude locations that are inactive for
    # the whole source year, but retain annually active locations when their
    # current-month series is zero. This keeps monthly component topology
    # stable without creating permanently inactive load components.
    added = 0
    zero_series = 0
    annual_zero = 0
    for bus, p_series in bus_load.items():
        if annual_bus_load[bus].abs().max() < 0.001:
            annual_zero += 1
            continue
        if p_series.abs().max() < 0.001:
            zero_series += 1
        load_name = bus
        if load_name not in network.loads.index:
            network.add("Load", load_name, bus=bus, p_set=0.0)
        # Always set timeseries via loads_t.p_set to ensure datetime index
        # is preserved in the exported CSV
        p_with_index = pd.Series(p_series.values, index=snapshots)
        if load_name in network.loads_t.p_set.columns:
            network.loads_t.p_set[load_name] = \
                network.loads_t.p_set[load_name] + p_with_index.values
        else:
            network.loads_t.p_set[load_name] = p_with_index
        added += 1

    log.info("    Direct loads: %d  Aggregated: %d  Skipped: %d  "
             "Total buses with load: %d",
             direct_count, agg_count, skip_count, added)
    if annual_zero:
        log.info("    Excluded annual-zero load locations: %d", annual_zero)
    if zero_series:
        log.info("    Retained zero-series load components: %d", zero_series)
    log.info("    Total mean load: %.1f MW",
             sum(s.mean() for s in bus_load.values()))
    return network


def make_snapshots(year: int, month: Optional[int], freq: str) -> pd.DatetimeIndex:
    """
    Build a DatetimeIndex for the given year (and optionally month).

    Parameters
    ----------
    year : int
    month : int or None
        If None, returns full year.  If int (1-12), returns that month only.
    freq : str
        Pandas offset string (e.g. '30min', '1h', '1D').

    Returns
    -------
    pd.DatetimeIndex
    """
    if month is None:
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        start = pd.Timestamp(year=year, month=month, day=1)
        if month == 12:
            end = pd.Timestamp(year=year + 1, month=1, day=1)
        else:
            end = pd.Timestamp(year=year, month=month + 1, day=1)

    return pd.date_range(start=start, end=end, freq=freq, inclusive="left")


# =============================================================================
# TOP-LEVEL NETWORK BUILDER
# =============================================================================

def build_base_network(
    config: Dict,
    paths: Dict,
    year: int,
    month: Optional[int] = None,
    island: Optional[str] = None,
) -> pypsa.Network:
    """
    Build a complete base PyPSA network for the given year and period.

    Parameters
    ----------
    config : dict
        Loaded YAML configuration.
    paths : dict
        Resolved Path objects from resolve_paths().
    year : int
        Data year (2024, 2025, ...).
    month : int or None
        Month number (1-12) for a monthly network, or None for annual.
    island : str or None
        'NI', 'SI', or None (full country).

    Returns
    -------
    pypsa.Network
    """
    period = f"{MONTHS[month-1]}_{year}" if month else str(year)
    log.info("\n%s", "=" * 60)
    log.info("BUILDING BASE NETWORK  year=%d  period=%s  island=%s",
             year, period, island or "full")
    log.info("%s", "=" * 60)

    freq = config.get("frequency", "30min")
    snapshot_hours = fixed_interval_hours(freq)
    timeseries_contract = annual_timeseries_contract(config)

    # ------------------------------------------------------------------
    # Load static data
    # ------------------------------------------------------------------
    log.info("\nLoading static network data")
    df_nodes = load_nodes(paths["nodes"])

    # Select lines file based on island scope.
    # SI-only build uses lines_data_si.csv; NI-only uses lines_data_ni.csv;
    # full-country build uses lines_data.csv (combined, when available).
    if island == "SI" and paths["lines_si"].exists():
        lines_path = paths["lines_si"]
        log.info("  Island='SI': using %s", lines_path.name)
    elif island == "NI" and paths["lines_ni"].exists():
        lines_path = paths["lines_ni"]
        log.info("  Island='NI': using %s", lines_path.name)
    else:
        lines_path = paths["lines"]
        log.info("  Using combined lines file: %s", lines_path.name)

    df_lines = load_lines(lines_path)
    df_links = load_links(paths["links"])
    df_gen   = load_generator_registry(paths["gen_reg"])

    # Detect multi-voltage substations from line endpoint data
    log.info("\nDetecting multi-voltage substations")
    multi_volt_sites = find_multi_voltage_sites(df_lines)

    # Identify DC bus sites from links data
    dc_sites = get_dc_node_sites(df_links)
    log.info("  DC bus sites: %s", sorted(dc_sites))

    # ------------------------------------------------------------------
    # Load timeseries
    # ------------------------------------------------------------------
    log.info("\nLoading timeseries data")
    df_export, df_import = load_annual_timeseries(
        paths["annual"], year,
        freq_source=timeseries_contract["source_interval"],
        freq_target=freq,
        input_quantity=timeseries_contract["input_quantity"],
        input_unit=timeseries_contract["input_unit"],
    )

    # ------------------------------------------------------------------
    # Build snapshots
    # ------------------------------------------------------------------
    snapshots = make_snapshots(year, month, freq)
    log.info("\nSnapshots: %d x %s  [%s -> %s]",
             len(snapshots), freq,
             snapshots[0].isoformat(), snapshots[-1].isoformat())

    # ------------------------------------------------------------------
    # Construct network
    # ------------------------------------------------------------------
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.snapshot_weightings.loc[:, ["objective", "stores", "generators"]] = (
        snapshot_hours
    )

    network = add_carriers(network)
    network = add_buses(network, df_nodes, df_lines, multi_volt_sites, dc_sites, island)
    network = add_lines(network, df_lines, multi_volt_sites, config=config)
    network = add_transformers(network, multi_volt_sites, df_nodes)
    network = add_hvdc_links(network, df_links, island)
    network = add_generators(
        network,
        df_gen,
        df_lines,
        multi_volt_sites,
        island=island,
    )
    network = add_loads(network, df_export, snapshots, df_nodes, config=config, island=island)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("\n%s", "=" * 60)
    log.info("NETWORK SUMMARY  %s", period)
    log.info("%s", "=" * 60)
    log.info("  Buses        : %d", len(network.buses))
    log.info("  Lines        : %d", len(network.lines))
    log.info("  Transformers : %d", len(network.transformers))
    log.info("  Links (HVDC) : %d", len(network.links))
    log.info("  Generators   : %d", len(network.generators))
    log.info("  Loads        : %d", len(network.loads))
    log.info("  Snapshots    : %d", len(network.snapshots))

    return network


# =============================================================================
# SAVE / LOAD UTILITIES
# =============================================================================

def save_network(
    network: pypsa.Network,
    paths: Dict,
    scenario: str,
    year: int,
    month: Optional[int] = None,
    island: Optional[str] = None,
) -> Path:
    """
    Save network to a CSV folder following the established naming convention.

    Folder structure:
        {networks}/{scenario}/{year}/{island_prefix}{period}/

    Parameters
    ----------
    network : pypsa.Network
    paths : dict
    scenario : str
    year : int
    month : int or None
    island : str or None

    Returns
    -------
    Path
        Path to the saved network folder.
    """
    period = f"{MONTHS[month-1]}_{year}" if month else str(year)
    prefix = f"{island.lower()}_" if island else ""
    folder = paths["networks"] / scenario / str(year) / f"{prefix}{period}"
    folder.mkdir(parents=True, exist_ok=True)

    # Calculate type-dependent impedance values before saving. Line s_nom is
    # assigned explicitly during construction because it is an optimisation
    # input, not a value derived from the line type by this method.
    network.calculate_dependent_values()
    network.export_to_csv_folder(str(folder))

    # Rewrite loads-p_set.csv with datetime index.
    # PyPSA 1.0.3 exports with integer index; we overwrite with
    # datetime strings so downstream scripts can reindex correctly.
    pset_path = folder / "loads-p_set.csv"
    if pset_path.exists() and not network.loads_t.p_set.empty:
        df_pset = network.loads_t.p_set.copy()
        df_pset.index = network.snapshots
        df_pset.to_csv(str(pset_path))

    log.info("  Saved to: %s", folder)
    return folder


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Build the three base-network representations for all configured
    years and months:

        SI   - South Island only
        NI   - North Island only
        NZ   - Combined North + South Island network, including HVDC

    The transmission source files are:

        lines_data_si.csv -> South Island network
        lines_data_ni.csv -> North Island network
        lines_data.csv    -> Combined NZ network

    These source files are treated as independent authoritative datasets.
    nza_base_net.py does not create or overwrite them.
    """

    # ------------------------------------------------------------------
    # Locate workspace
    # ------------------------------------------------------------------
    analysis_root = Path(__file__).resolve().parents[1]
    misc_dir = analysis_root / "misc"
    nza_root_file = misc_dir / "nza_root.py"

    if not nza_root_file.exists():
        raise FileNotFoundError(
            f"Cannot locate nza_root.py: {nza_root_file}"
        )

    misc_dir_str = str(misc_dir)

    if misc_dir_str not in sys.path:
        sys.path.insert(0, misc_dir_str)

    from nza_root import WORKSPACE

    config_path = WORKSPACE / "config" / "nza_net_config.yaml"

    log.info("Workspace: %s", WORKSPACE)
    log.info("Config   : %s", config_path)

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------
    config = load_config(config_path)
    paths = resolve_paths(config)

    scenario = config.get("scenario", "reference")
    frequency = config.get("frequency", "30min")

    start_year = int(config["scenario_years"]["start"])
    end_year = int(config["scenario_years"]["end"])
    years = list(range(start_year, end_year + 1))

    log.info("Scenario : %s", scenario)
    log.info("Years    : %s", years)
    log.info("Frequency: %s", frequency)

    # ------------------------------------------------------------------
    # Verify all three transmission datasets exist
    # ------------------------------------------------------------------
    required_line_files = {
        "South Island": paths["lines_si"],
        "North Island": paths["lines_ni"],
        "Combined NZ": paths["lines"],
    }

    missing_files = [
        f"{name}: {path}"
        for name, path in required_line_files.items()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required transmission dataset(s) missing:\n  "
            + "\n  ".join(missing_files)
        )

    log.info("Transmission datasets:")
    log.info("  SI       : %s", paths["lines_si"])
    log.info("  NI       : %s", paths["lines_ni"])
    log.info("  Combined : %s", paths["lines"])

    # ------------------------------------------------------------------
    # Validate line identifiers in each source dataset
    # ------------------------------------------------------------------
    for label, path in required_line_files.items():
        df = pd.read_csv(path, dtype=str)

        if "line" not in df.columns:
            raise ValueError(
                f"{label} transmission file has no 'line' column: {path}"
            )

        line_ids = df["line"].fillna("").str.strip()

        blank_ids = line_ids.eq("")

        if blank_ids.any():
            csv_rows = list(df.index[blank_ids] + 2)

            raise ValueError(
                f"{label} transmission file contains blank line "
                f"identifiers at CSV rows: {csv_rows}"
            )

        duplicate_ids = line_ids[line_ids.duplicated()].unique()

        if len(duplicate_ids):
            raise ValueError(
                f"{label} transmission file contains duplicate line "
                f"identifiers: {list(duplicate_ids)}"
            )

        log.info(
            "  Validated %-11s: %d transmission lines",
            label,
            len(df),
        )

    # ------------------------------------------------------------------
    # Build all three network representations
    #
    # island="SI" -> lines_data_si.csv
    # island="NI" -> lines_data_ni.csv
    # island=None -> lines_data.csv + HVDC link
    # ------------------------------------------------------------------
    networks_to_build = [
        ("SI", "South Island"),
        ("NI", "North Island"),
        (None, "Combined NZ"),
    ]

    for year in years:
        for month_num in range(1, 13):

            for island, label in networks_to_build:

                log.info(
                    "Building %s %d/%02d [%s]",
                    scenario,
                    year,
                    month_num,
                    label,
                )

                try:
                    network = build_base_network(
                        config=config,
                        paths=paths,
                        year=year,
                        month=month_num,
                        island=island,
                    )

                    save_network(
                        network=network,
                        paths=paths,
                        scenario=scenario,
                        year=year,
                        month=month_num,
                        island=island,
                    )

                except Exception:
                    log.exception(
                        "Failed building %s %d/%02d [%s]",
                        scenario,
                        year,
                        month_num,
                        label,
                    )

                    # Do not silently continue and leave an apparently
                    # complete but partially failed network dataset.
                    raise

    log.info("=" * 60)
    log.info("Base-network build complete.")
    log.info("Built SI, NI and combined NZ networks.")
    log.info("Next: run nza_custom_net.py to add profiles and costs.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
