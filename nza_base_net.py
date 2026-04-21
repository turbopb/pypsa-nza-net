#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nza_base_net.py

Build PyPSA network topology and load timeseries for NZ capacity expansion.

This script is part of the pypsa-nza-net module.  It reads clean, pre-processed
input data produced by pypsa-nza-data and constructs a PyPSA network with:

    - Buses from nodes.csv (110 kV and above AC; dedicated DC buses)
    - AC transmission lines from lines_data.csv (110 kV and 220 kV only)
    - Transformers at multi-voltage substations (auto-detected)
    - Cook Strait HVDC link from links_data.csv (as PyPSA Link)
    - Generators from generator_registry.csv (all dormant at p_max_pu=0)
    - Load timeseries from {year}_export_sites_all.csv

Island construction
-------------------
North Island, South Island, or full country networks can be built
independently.  The HVDC link is only added when building the full country
network (island=None).

Timestep resampling
-------------------
Source data is 30-minute half-hourly.  A target timestep can be specified
in the YAML config ('frequency' key, pandas offset string).  Resampling is
applied to all timeseries before network construction.

Network output
--------------
Networks are saved as PyPSA CSV folders under:
    {root}/models/networks/{scenario}/{year}/

Author: Phillippe Bruneau
"""

import sys
import logging
import calendar
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
    Return the set of all bus site codes that appear in links_data.csv.

    These sites are DC infrastructure nodes and must be added as 350 kV DC
    buses regardless of what nodes.csv says about their voltage (some
    intermediate nodes such as OTB have no voltage entry in nodes.csv).

    Parameters
    ----------
    df_links : pd.DataFrame

    Returns
    -------
    set of str
    """
    dc_sites = set()
    for _, row in df_links.iterrows():
        if str(row.get("carrier", "")).upper() == "DC":
            dc_sites.add(str(row["bus0"]).strip())
            dc_sites.add(str(row["bus1"]).strip())
    return dc_sites


# Minimum AC voltage (kV) for inclusion in the transmission network.
# Buses below this threshold can still carry load and generation but
# will not be endpoints of AC transmission lines.
AC_VOLT_THRESHOLD_KV = 110

# PyPSA standard transformer type for 220/110 kV inter-bus transformers.
# s_nom is set extendable with capital_cost=0 as placeholder.
TRANSFORMER_TYPE = "630MVA 220/110kV"

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

    df["volts"]    = pd.to_numeric(df["volts"],    errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load annual export (load) and import (generation) timeseries.

    Resamples from source frequency to target frequency if they differ.
    Export values are energy in MWh per half-hour; import likewise.
    When resampling to a coarser resolution, sum (energy) is used.

    Parameters
    ----------
    annual_dir : Path
        Directory containing {year}_export_sites_all.csv etc.
    year : int
    freq_source : str
        Native data frequency (default '30min').
    freq_target : str
        Target frequency for the network (e.g. '30min', '1h', '1D').

    Returns
    -------
    (df_export, df_import) : tuple of pd.DataFrame
        Both indexed by datetime, columns are site codes.
    """
    export_path = annual_dir / f"{year}_export_sites_all.csv"
    import_path = annual_dir / f"{year}_import_sites_all.csv"

    for p in (export_path, import_path):
        if not p.exists():
            raise FileNotFoundError(f"Timeseries file not found: {p}")

    df_export = pd.read_csv(export_path, index_col=0, parse_dates=True)
    df_import = pd.read_csv(import_path, index_col=0, parse_dates=True)

    log.info("  Export: %d timesteps x %d sites", len(df_export), len(df_export.columns))
    log.info("  Import: %d timesteps x %d sites", len(df_import), len(df_import.columns))

    if freq_target != freq_source:
        log.info("  Resampling %s -> %s (sum for energy)", freq_source, freq_target)
        df_export = df_export.resample(freq_target).sum()
        df_import = df_import.resample(freq_target).sum()
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

    added = 0
    skipped = 0

    for _, row in df.iterrows():
        site    = row["site"]
        volts   = row["volts"]   # may be NaN for junction nodes
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

        # AC buses
        carrier = "AC"
        v_nom = float(volts) if pd.notna(volts) else 0.0

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
) -> pypsa.Network:
    """
    Add AC transmission lines to the network.

    Line electrical parameters (r, x, b) are computed by PyPSA from the
    standard line type library and the circuit length.  The 'distance'
    column from lines_data.csv is used as the circuit length in km.

    Double-circuit lines (config == 'double') are represented with
    num_parallel=2.

    Parameters
    ----------
    network : pypsa.Network
    df_lines : pd.DataFrame
    multi_volt_sites : dict

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

        network.add(
            "Line",
            line_name,
            bus0=bus0,
            bus1=bus1,
            length=length_km,
            type=line_type,
            num_parallel=num_parallel,
            carrier="AC",
            s_nom_extendable=True,
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
            s_nom_extendable=True,
            capital_cost=0.0,   # placeholder: update in YAML cost file
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
    Add HVDC links (Cook Strait) to the network.

    Links are added only when building the full-country network (island=None).
    The link connects BEN-DC to HAY-DC regardless of what bus0/bus1 say in
    the CSV, correcting for the historical BEN/HAY AC bus naming error.

    Efficiency is read from the CSV if present; otherwise derived from a
    standard 0.011 Ohm/km DC cable resistance at rated conditions, giving
    approximately 0.965 for the 552 km Cook Strait cable.

    Parameters
    ----------
    network : pypsa.Network
    df_links : pd.DataFrame
    island : str or None

    Returns
    -------
    pypsa.Network
    """
    if island is not None:
        log.info("\n  HVDC links: skipped (single-island build)")
        return network

    log.info("\n  Adding HVDC links")

    # Bus names are used directly from links_data.csv.
    # The CSV uses the correct DC bus names throughout:
    #   BEN-DC, FTB, OTB, HAY-DC
    # No remapping is required.

    added = 0
    for _, row in df_links.iterrows():
        link_name = row["link"]
        bus0      = str(row["bus0"]).strip()
        bus1      = str(row["bus1"]).strip()
        length_km = row["length"]

        if bus0 not in network.buses.index:
            log.warning("    Link %s: bus0 '%s' not in network -- skipped",
                        link_name, bus0)
            continue
        if bus1 not in network.buses.index:
            log.warning("    Link %s: bus1 '%s' not in network -- skipped",
                        link_name, bus1)
            continue

        # Efficiency: use CSV value if available, otherwise derive from length
        if "efficiency" in df_links.columns and pd.notna(row.get("efficiency")):
            efficiency = float(row["efficiency"])
        else:
            # Standard DC cable: 0.011 Ohm/km, ~3.5% losses at 552 km
            r_total   = 0.011 * float(length_km)
            efficiency = round(1.0 - (r_total / 600.0), 4)   # approximate
            log.info("    %s: efficiency derived from cable resistance: %.4f",
                     link_name, efficiency)

        # p_nom from CSV or default to 1200 MW (Cook Strait rated capacity)
        p_nom = 1200.0

        network.add(
            "Link",
            link_name,
            bus0=bus0,
            bus1=bus1,
            length=float(length_km),
            carrier="DC",
            p_max_pu=1.0,
            p_min_pu=-1.0,
            p_nom=p_nom,
            efficiency=efficiency,
            marginal_cost=0.0,
            capital_cost=0.0,
            p_nom_extendable=False,
            build_year=int(row.get("build_year", 1980)),
        )
        log.info("    Added: %s  %s -> %s  %.0f MW  eta=%.4f",
                 link_name, bus0, bus1, p_nom, efficiency)
        added += 1

    log.info("    Added %d HVDC links", added)
    return network


def add_generators(
    network: pypsa.Network,
    df_gen: pd.DataFrame,
    multi_volt_sites: Dict,
) -> pypsa.Network:
    """
    Add generators to the network from the generator registry.

    All generators are added dormant (p_max_pu=0).  They are activated
    by nza_custom_net.py during the customisation step.

    Parameters
    ----------
    network : pypsa.Network
    df_gen : pd.DataFrame
        Generator registry from build_generator_registry.py.
    multi_volt_sites : dict

    Returns
    -------
    pypsa.Network
    """
    log.info("\n  Adding generators (all dormant)")

    added   = 0
    skipped = 0

    for _, row in df_gen.iterrows():
        bus_name = row["bus"]   # already resolved in generator_registry.csv

        if bus_name not in network.buses.index:
            log.warning("    Generator at bus '%s' not in network -- skipped", bus_name)
            skipped += 1
            continue

        carrier    = row["carrier"]
        gen_name   = f"{bus_name}_{carrier}"
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
) -> pypsa.Network:
    """
    Add load timeseries to the network.

    Only sites present in both df_export and the network bus list receive
    a load component.  The snapshot index is used to slice the annual
    timeseries to the correct period.

    Parameters
    ----------
    network : pypsa.Network
    df_export : pd.DataFrame
        Annual export timeseries (MWh per interval, columns=site codes).
    snapshots : pd.DatetimeIndex
        Snapshot index for this network period.

    Returns
    -------
    pypsa.Network
    """
    log.info("\n  Adding loads")

    # Align export data to snapshots
    df_slice = df_export.reindex(snapshots)

    added   = 0
    skipped = 0

    for site in df_slice.columns:
        if site not in network.buses.index:
            skipped += 1
            continue

        p_set = df_slice[site].tolist()

        if len(p_set) != len(snapshots):
            log.warning("    Load %s: length mismatch %d vs %d -- skipped",
                        site, len(p_set), len(snapshots))
            skipped += 1
            continue

        network.add("Load", site, bus=site, p_set=p_set)
        added += 1

    log.info("    Added %d loads (%d sites not in network)", added, skipped)
    return network


# =============================================================================
# SNAPSHOT CONSTRUCTION
# =============================================================================

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
        start = f"{year}-01-01 00:00"
        end   = f"{year}-12-31 23:30"
    else:
        last_day = calendar.monthrange(year, month)[1]
        start = f"{year}-{month:02d}-01 00:00"
        end   = f"{year}-{month:02d}-{last_day:02d} 23:30"

    return pd.date_range(start=start, end=end, freq=freq)


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

    # ------------------------------------------------------------------
    # Load static data
    # ------------------------------------------------------------------
    log.info("\nLoading static network data")
    df_nodes = load_nodes(paths["nodes"])
    df_lines = load_lines(paths["lines"])
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
        freq_source="30min",
        freq_target=freq,
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

    network = add_carriers(network)
    network = add_buses(network, df_nodes, multi_volt_sites, dc_sites, island)
    network = add_lines(network, df_lines, multi_volt_sites)
    network = add_transformers(network, multi_volt_sites, df_nodes)
    network = add_hvdc_links(network, df_links, island)
    network = add_generators(network, df_gen, multi_volt_sites)
    network = add_loads(network, df_export, snapshots)

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
    network.export_to_csv_folder(str(folder))
    log.info("  Saved to: %s", folder)
    return folder


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Build base networks for all configured years and months.

    Reads configuration from the default YAML path.  Builds monthly networks
    for the base year and any future years defined in the config.
    """
    # Locate workspace and config
    # Locate workspace via nza_root (reads PYPSA_NZA_ROOT env var).
    # nza_root.py must sit in the project root directory alongside all
    # module repos.  See SETUP.md for setup instructions.
    _root_str = __import__('os').environ.get('PYPSA_NZA_ROOT', '')
    if _root_str:
        sys.path.insert(0, str(Path(_root_str).resolve()))
    from nza_root import WORKSPACE
    config_path = WORKSPACE / "config" / "nza_net_config.yaml"

    log.info("Config: %s", config_path)
    config = load_config(config_path)
    paths  = resolve_paths(config)

    scenario  = config.get("scenario", "reference")
    base_year = config.get("base_year", 2024)
    freq      = config.get("frequency", "30min")

    start_year = config["scenario_years"]["start"]
    end_year   = config["scenario_years"]["end"]
    years      = list(range(start_year, end_year + 1))

    log.info("Scenario : %s", scenario)
    log.info("Years    : %s", years)
    log.info("Frequency: %s", freq)

    for year in years:
        for month_num in range(1, 13):
            try:
                network = build_base_network(
                    config=config,
                    paths=paths,
                    year=year,
                    month=month_num,
                    island=None,   # full country
                )
                save_network(network, paths, scenario, year, month_num)
            except Exception as exc:
                log.error("Error building %s %d/%d: %s",
                          scenario, year, month_num, exc)
                continue

    log.info("\nDone.  Run nza_custom_net.py to add profiles and costs.")


if __name__ == "__main__":
    main()
