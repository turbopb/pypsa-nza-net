#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nza_custom_net.py

Customise PyPSA networks for capacity expansion analysis.

Reads base networks produced by nza_base_net.py and adds:
    - Generator availability (p_max_pu): scalar for dispatchable plant,
      timeseries for wind and solar.
    - Brownfield expansion limits for existing renewable sites.
    - Greenfield generation options (new candidate sites).
    - Technology costs (annualised capital and marginal).

Workflow
--------
    nza_base_net.py  -->  base networks (topology + loads, dormant generators)
    nza_custom_net.py -->  customised networks (ready for optimisation)

Output networks are saved back to the same CSV folder, overwriting the base
network.  Run nza_cx_opt.py next to optimise capacity expansion.

Author: Phillippe Bruneau
"""

import sys
import logging
import calendar
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
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

# Custom transformer types not in the PyPSA standard library.
# Must match the type name used in nza_base_net.py.
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


# =============================================================================
# MONTHS
# =============================================================================

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_paths(config: Dict) -> Dict:
    """Build absolute Path objects for all data inputs and outputs."""
    root = Path(config["paths"]["root"])
    return {
        "networks":    root / config["paths"]["dirpath_networks"],
        "annual":      root / config["paths"]["dirpath_annual"],
        "static":      root / config["paths"]["dirpath_static"],
        "renewables":  root / config["paths"]["dirpath_renewables"],
        # wind_mon_root: root of monthly wind profile directory
        # Full path per year is built at runtime in process_monthly_network
        "wind_mon_root": root / config["paths"]["dirpath_renewables"]
                              / "wind" / "wind_p_pu_monthly",
        "solar":         root / config["paths"]["dirpath_renewables"]
                              / config["paths"]["dirpath_solar_pu"],
        "costs":       root / config["paths"]["dirpath_costs"],
        "gen_reg":     root / config["paths"]["dirpath_static"]
                            / config["filename"]["generator_registry"],
    }


# =============================================================================
# ACTIVE GENERATOR DETECTION
# =============================================================================

def get_active_generators(
    network: pypsa.Network,
    import_path: Path,
    year: int,
    month_num: int,
) -> List[str]:
    """
    Return list of generator names active in this network period.

    A generator is considered active if:
      - It exists in the network, AND
      - Its site appears as a column with non-zero values in the annual
        grid import timeseries for the relevant month.

    This replaces the old approach of reading monthly {YYYYMM}_g_MW.csv
    files, which no longer exist in the new pipeline.

    Parameters
    ----------
    network : pypsa.Network
    import_path : Path
        Directory containing {year}_import_sites_all.csv.
    year : int
    month_num : int
        Month number 1-12.

    Returns
    -------
    list of str
        Generator names (as stored in network.generators.index).
    """
    import_file = import_path / f"{year}_import_sites_all.csv"
    if not import_file.exists():
        log.warning("Import timeseries not found: %s -- all generators inactive",
                    import_file)
        return []

    df = pd.read_csv(import_file, index_col=0, parse_dates=True)

    # Slice to this month
    df_month = df[df.index.month == month_num]

    # Sites with any non-zero import in this month
    active_sites = set(
        col for col in df_month.columns
        if df_month[col].sum() > 0
    )

    # Match to generator bus names in network.
    # Generator names follow {bus}_{carrier} convention from nza_base_net.py.
    active_gens = []
    for gen_name in network.generators.index:
        bus = network.generators.at[gen_name, "bus"]
        # Bus name may be suffixed (e.g. ROX_110) -- extract site code
        site = bus.split("_")[0] if "_" in bus else bus
        if site in active_sites:
            active_gens.append(gen_name)

    log.info("    Active generators this month: %d / %d",
             len(active_gens), len(network.generators))
    return active_gens


# =============================================================================
# GENERATOR AVAILABILITY
# =============================================================================

def set_dispatchable_availability(
    network: pypsa.Network,
    generator_list: List[str],
    availability: Dict[str, float],
    month_num: int = 0,
    hydro_seasonal: Dict[int, float] = None,
) -> pypsa.Network:
    """
    Set scalar p_max_pu for dispatchable generators.

    Uses the availability map from the YAML config. If hydro_seasonal is
    provided, hydro p_max_pu varies by month number (1=Jan..12=Dec).
    Wind and solar are skipped here -- handled by timeseries profiles.

    Parameters
    ----------
    network : pypsa.Network
    generator_list : list of str
        Active generator names.
    availability : dict
        Carrier -> scalar availability (from config['availability']).
    month_num : int
        Month number (1-12). Used for seasonal hydro availability.
    hydro_seasonal : dict or None
        {month_num: p_max_pu} seasonal profile for hydro.

    Returns
    -------
    pypsa.Network
    """
    renewable_carriers = {"wind", "solar_pv", "off_wind"}

    for gen in generator_list:
        if gen not in network.generators.index:
            log.warning("    Generator '%s' not found in network -- skipped", gen)
            continue

        carrier = network.generators.at[gen, "carrier"]

        if carrier in renewable_carriers:
            continue   # handled by timeseries assignment

        # Use seasonal hydro profile if available
        if carrier == "hydro" and hydro_seasonal and month_num in hydro_seasonal:
            network.generators.at[gen, "p_max_pu"] = float(hydro_seasonal[month_num])
        elif carrier in availability:
            network.generators.at[gen, "p_max_pu"] = availability[carrier]
        else:
            log.warning("    No availability entry for carrier '%s' (gen %s)",
                        carrier, gen)

    return network


# =============================================================================
# RENEWABLE PROFILES
# =============================================================================

def assign_renewable_profiles(
    network: pypsa.Network,
    profiles_df: pd.DataFrame,
    carrier: str,
    profile_mapping: Dict[str, str],
) -> pypsa.Network:
    """
    Assign p_max_pu timeseries to renewable generators (wind or solar).

    Parameters
    ----------
    network : pypsa.Network
    profiles_df : pd.DataFrame
        Capacity factor profiles indexed by datetime, columns = site codes.
    carrier : str
        Carrier to match (e.g. 'wind', 'solar_pv').
    profile_mapping : dict
        {site: proxy_site} for sites without direct profile data.

    Returns
    -------
    pypsa.Network
    """
    n_snaps = len(network.snapshots)
    generators = network.generators[
        network.generators["carrier"] == carrier
    ].index

    for gen in generators:
        bus     = network.generators.at[gen, "bus"]
        site    = bus.split("_")[0] if "_" in bus else bus
        profile_site = profile_mapping.get(site, site)

        if profile_site not in profiles_df.columns:
            log.warning("    %s: no profile found (checked '%s') -- skipped",
                        gen, profile_site)
            continue

        profile = profiles_df[profile_site].reindex(network.snapshots)

        if profile.isna().all():
            log.warning("    %s: profile has no overlap with network snapshots",
                        gen)
            continue

        # Fill any residual NaN at edges with 0 (boundary timesteps)
        profile = profile.fillna(0.0)

        if site != profile_site:
            log.info("    %s at %s: using proxy profile from %s", gen, site, profile_site)

        # Assign as explicit timeseries with correct index alignment
        network.generators_t.p_max_pu[gen] = pd.Series(
            profile.values,
            index=network.snapshots,
            dtype=float,
        )

    return network


# =============================================================================
# BROWNFIELD EXPANSION LIMITS
# =============================================================================

def set_brownfield_expansion_limits(
    network: pypsa.Network,
    carrier: str,
    expansion_factors: Dict[str, float],
) -> pypsa.Network:
    """
    Set p_nom expansion limits for existing generators (brownfield).

    Parameters
    ----------
    network : pypsa.Network
    carrier : str
    expansion_factors : dict
        {site_code: factor} -- existing capacity can grow by this multiple.

    Returns
    -------
    pypsa.Network
    """
    for gen in network.generators.index:
        if "_greenfield" in gen:
            continue
        if network.generators.at[gen, "carrier"] != carrier:
            continue

        bus  = network.generators.at[gen, "bus"]
        site = bus.split("_")[0] if "_" in bus else bus

        if site not in expansion_factors:
            continue

        p_nom = network.generators.at[gen, "p_nom"]
        factor = expansion_factors[site]

        network.generators.at[gen, "p_nom_extendable"] = True
        network.generators.at[gen, "p_nom_min"]        = p_nom
        network.generators.at[gen, "p_nom_max"]        = p_nom * factor

    return network


# =============================================================================
# GREENFIELD GENERATION
# =============================================================================

def get_technology_cost(cost_dict: Dict, year: int) -> float:
    """
    Return technology cost for the given year, interpolating between entries.

    Parameters
    ----------
    cost_dict : dict
        {year: cost} mapping.
    year : int

    Returns
    -------
    float
    """
    if not cost_dict:
        return 250000.0   # fallback: NZD 250k/MW

    if len(cost_dict) == 1:
        return float(list(cost_dict.values())[0])

    years = sorted(cost_dict.keys())
    costs = [cost_dict[y] for y in years]

    if year <= years[0]:
        return float(costs[0])
    if year >= years[-1]:
        return float(costs[-1])

    return float(np.interp(year, years, costs))


def add_greenfield_generators(
    network: pypsa.Network,
    config: Dict,
    year: int,
) -> pypsa.Network:
    """
    Add greenfield generation candidates (new sites the optimiser can build).

    Skips generators that already exist to avoid overwriting capacity values
    carried forward from a previous planning horizon.

    Parameters
    ----------
    network : pypsa.Network
    config : dict
    year : int

    Returns
    -------
    pypsa.Network
    """
    log.info("  Adding greenfield generation options")

    greenfield = config.get("greenfield_sites", {})
    tech_costs = config.get("technology_costs", {})

    if not greenfield:
        log.info("    No greenfield sites defined in config")
        return network

    added   = 0
    skipped = 0

    # --- Wind ---
    for site in greenfield.get("wind", []):
        gen_name = f"{site['name']}_greenfield"
        if gen_name in network.generators.index:
            skipped += 1
            continue

        capital_cost = get_technology_cost(tech_costs.get("wind", {}), year)

        network.add(
            "Generator", gen_name,
            bus=site["bus"],
            p_nom=0.0,
            p_nom_extendable=True,
            p_nom_min=0.0,
            p_nom_max=site["max_capacity"],
            carrier="wind",
            capital_cost=capital_cost,
            marginal_cost=12.0,
            efficiency=1.0,
            p_max_pu=0.0,
            build_year=year,
        )
        log.info("    Added wind greenfield: %s (max %.0f MW)", gen_name, site["max_capacity"])
        added += 1

    # --- Solar ---
    for site in greenfield.get("solar", []):
        gen_name = f"{site['name']}_greenfield"
        if gen_name in network.generators.index:
            skipped += 1
            continue

        capital_cost = get_technology_cost(tech_costs.get("solar", {}), year)

        network.add(
            "Generator", gen_name,
            bus=site["bus"],
            p_nom=0.0,
            p_nom_extendable=True,
            p_nom_min=0.0,
            p_nom_max=site["max_capacity"],
            carrier="solar_pv",
            capital_cost=capital_cost,
            marginal_cost=8.0,
            efficiency=1.0,
            p_max_pu=0.0,
            build_year=year,
        )
        log.info("    Added solar greenfield: %s (max %.0f MW)", gen_name, site["max_capacity"])
        added += 1

    # --- Battery ---
    for site in greenfield.get("battery", []):
        storage_name = f"{site['name']}_greenfield"
        if storage_name in network.storage_units.index:
            skipped += 1
            continue

        battery_cost = get_technology_cost(tech_costs.get("battery_4h", {}), year)

        network.add(
            "StorageUnit", storage_name,
            bus=site["bus"],
            p_nom=0.0,
            p_nom_extendable=True,
            p_nom_max=site["power"],
            carrier="battery",
            max_hours=4.0,
            capital_cost=battery_cost * 4 * 1000,
            marginal_cost=0.5,
            efficiency_store=0.95,
            efficiency_dispatch=0.95,
            cyclic_state_of_charge=True,
        )
        log.info("    Added battery greenfield: %s (%.0f MW / %.0f MWh)",
                 storage_name, site["power"], site["power"] * 4)
        added += 1

    log.info("    Greenfield: added %d, skipped %d (already exist)", added, skipped)
    return network


# =============================================================================
# TECHNOLOGY COSTS
# =============================================================================

def calculate_annuity(discount_rate: float, lifetime: int) -> float:
    """Annuity factor for capital cost calculation."""
    if discount_rate == 0:
        return 1.0 / lifetime
    return discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime)


def load_cost_data(cost_path: Path) -> Dict:
    """Load technology cost data from YAML file."""
    if not cost_path.exists():
        raise FileNotFoundError(f"Cost file not found: {cost_path}")
    with open(cost_path, "r") as f:
        return yaml.safe_load(f)


def apply_generator_costs(
    network: pypsa.Network,
    costs: Dict,
    year: int,
    discount_rate: float = 0.07,
) -> pypsa.Network:
    """
    Apply annualised capital and marginal costs to all generators.

    For extendable generators, the full annualised capital cost is applied.
    For non-extendable generators commissioned more than 10 years ago,
    capital cost is set to zero (asset considered written down).

    Parameters
    ----------
    network : pypsa.Network
    costs : dict
        Technology cost data from YAML.
    year : int
        Planning year (used for depreciation calculation).
    discount_rate : float

    Returns
    -------
    pypsa.Network
    """
    # Import standard_modality inline to avoid circular dependency
    def _std(modality: str) -> str:
        MODALITY_MAP = {
            "wind": "wind", "solar_pv": "solar_pv", "hydro": "hydro",
            "geothermal": "geothermal", "ocgt": "ocgt", "ccgt": "ccgt",
            "gas": "ocgt", "coal_gas": "coal_gas", "coal": "coal",
            "diesel": "diesel", "battery": "battery",
            "pumped_hydro": "pumped_hydro", "bioenergy": "bioenergy",
        }
        return MODALITY_MAP.get(modality.lower(), modality.lower())

    carriers = network.generators["carrier"].unique()

    for carrier in carriers:
        carrier_std = _std(carrier)

        if carrier_std not in costs:
            log.warning("    No cost data for '%s' -- skipping", carrier_std)
            continue

        cost_data        = costs[carrier_std]
        capital_invest   = cost_data.get("capital_cost_nzd_per_mw", 0.0)
        marginal_cost    = cost_data.get("marginal_cost_nzd_per_mwh", 0.0)
        lifetime         = cost_data.get("lifetime_years", 25)

        fom_rate = cost_data.get("fixed_om_cost_nzd_per_mw_year", 0.0)
        fom_abs  = fom_rate * capital_invest if fom_rate < 1 else fom_rate

        annuity            = calculate_annuity(discount_rate, lifetime)
        annualised_capital = annuity * capital_invest + fom_abs

        mask = network.generators["carrier"] == carrier

        for gen in network.generators[mask].index:
            is_extendable = network.generators.at[gen, "p_nom_extendable"]

            if is_extendable:
                gen_capital_cost = annualised_capital
            else:
                build_year = network.generators.at[gen, "build_year"]
                gen_capital_cost = 0.0 if (year - build_year) > 10 else annualised_capital

            # Write costs back to network -- this was the bug in the original
            network.generators.at[gen, "capital_cost"]  = gen_capital_cost
            network.generators.at[gen, "marginal_cost"] = marginal_cost

    return network


def setup_costs(
    network: pypsa.Network,
    paths: Dict,
    config: Dict,
    year: int,
) -> pypsa.Network:
    """Load cost YAML and apply to network."""
    cost_path = paths["costs"] / config["filename"]["costs_file"]
    costs     = load_cost_data(cost_path)
    network   = apply_generator_costs(network, costs, year)
    return network


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_monthly_network(
    month_num: int,
    year: int,
    config: Dict,
    paths: Dict,
) -> None:
    """
    Customise a single monthly network.

    Parameters
    ----------
    month_num : int
        Month number 1-12.
    year : int
    config : dict
    paths : dict
    """
    month_name   = MONTHS[month_num - 1]
    scenario     = config.get("scenario", "reference")

    # Process all existing network variants for this month:
    #   si_jan_2024  -- South Island standalone
    #   ni_jan_2024  -- North Island standalone
    #   jan_2024     -- Combined two-island network
    base_label = f"{month_name}_{year}"
    net_base   = paths["networks"] / scenario / str(year)

    variants = []
    for label in [f"si_{base_label}", f"ni_{base_label}", base_label]:
        if (net_base / label).exists():
            variants.append(label)

    if not variants:
        log.error("  No network folder found for %s/%s", year, base_label)
        return

    # Process first variant -- rest handled in loop at end of function
    month_label = variants[0]
    network_dir = net_base / month_label

    log.info("\n%s", "=" * 60)
    log.info("Customising  %s", month_label)
    log.info("%s", "=" * 60)

    if not network_dir.exists():
        log.error("  Network folder not found: %s", network_dir)
        return

    # Load network
    network = pypsa.Network()
    network.import_from_csv_folder(str(network_dir))

    # Restore standard component type libraries lost during CSV export/import.
    # PyPSA does not write built-in types (transformer_types, line_types) to
    # CSV -- they must be reloaded from a fresh network instance.
    _ref = pypsa.Network()
    network.transformer_types = _ref.transformer_types
    network.line_types        = _ref.line_types
    register_custom_transformer_types(network)

    network.consistency_check()

    availability      = config.get("availability", {})
    wind_mapping      = config.get("WIND_PROFILE_MAPPING", {})
    solar_mapping     = config.get("SOLAR_PROFILE_MAPPING", {})
    wind_exp_factors  = config.get("WIND_EXPANSION_FACTORS", {})
    solar_exp_factors = config.get("SOLAR_EXPANSION_FACTORS", {})
    weather_year      = config.get("weather_year", year)

    # --- Step 1: Determine active generators ---
    log.info("\n  Step 1: Identifying active generators")
    active_gens = get_active_generators(
        network, paths["annual"], year, month_num
    )

    # --- Step 2: Set dispatchable availability ---
    log.info("\n  Step 2: Setting dispatchable availability")
    hydro_seasonal = config.get("hydro_seasonal_availability", {})
    # Convert string keys to int (YAML may load them as strings)
    hydro_seasonal = {int(k): float(v) for k, v in hydro_seasonal.items()}
    network = set_dispatchable_availability(
        network, active_gens, availability,
        month_num=month_num,
        hydro_seasonal=hydro_seasonal,
    )
    if hydro_seasonal:
        hydro_pu = hydro_seasonal.get(month_num, availability.get("hydro", 0.75))
        log.info("    Hydro p_max_pu this month (month %d): %.2f", month_num, hydro_pu)

    # --- Step 3: Wind profiles ---
    log.info("\n  Step 3: Assigning wind profiles")
    # Wind profiles are stored per simulation year.
    # File: wind_p_pu_monthly/{year}/wind_p_pu_{YYYYMM}.csv
    # Timestamps are relabelled to simulation year by generate_wind_profiles.py
    month_code = f"{year}{month_num:02d}"
    wind_path  = paths["wind_mon_root"] / str(year) / f"wind_p_pu_{month_code}.csv"

    if wind_path.exists():
        wind_df = pd.read_csv(wind_path, index_col=0, parse_dates=True)
        network = assign_renewable_profiles(network, wind_df, "wind", wind_mapping)
        network = set_brownfield_expansion_limits(network, "wind", wind_exp_factors)
    else:
        log.warning("    Wind profile not found: %s", wind_path)
        log.warning("    Run generate_wind_profiles.py --years %d first", year)

    # --- Step 4: Solar profiles ---
    log.info("\n  Step 4: Assigning solar profiles")
    # For planning years use year-specific file; fall back to base-year alias.
    solar_year_file  = f"solar_PV_p_pu_{year}.csv"
    solar_alias_file = config["filename"].get("solar", "solar_PV_p_pu.csv")
    solar_path = paths["solar"] / solar_year_file
    if not solar_path.exists():
        solar_path = paths["solar"] / solar_alias_file
        log.info("    Year-specific solar file not found -- using alias: %s",
                 solar_alias_file)

    if solar_path.exists():
        solar_df = pd.read_csv(solar_path, index_col=0, parse_dates=True)
        solar_df.index = pd.to_datetime(solar_df.index)
        solar_df = solar_df[~solar_df.index.duplicated(keep="first")]
        network  = assign_renewable_profiles(network, solar_df, "solar_pv", solar_mapping)
        network  = set_brownfield_expansion_limits(network, "solar_pv", solar_exp_factors)
    else:
        log.warning("    Solar profile not found: %s", solar_path)
        log.warning("    Run generate_solar_profiles.py --years %d first", year)

    # --- Step 5: Greenfield options ---
    log.info("\n  Step 5: Adding greenfield options")
    network = add_greenfield_generators(network, config, year)

    # --- Step 6: Technology costs ---
    log.info("\n  Step 6: Applying technology costs")
    network = setup_costs(network, paths, config, year)

    # --- Step 7: Save ---
    log.info("\n  Step 7: Saving")
    network.export_to_csv_folder(str(network_dir))
    log.info("    Saved to %s", network_dir)

    # --- Process remaining variants (ni_, combined) ---
    for extra_label in variants[1:]:
        extra_dir = net_base / extra_label
        log.info("")
        log.info("=" * 60)
        log.info("Customising  %s", extra_label)
        log.info("=" * 60)
        try:
            extra_net = pypsa.Network()
            extra_net.import_from_csv_folder(str(extra_dir))
            # Restore standard component type libraries
            _ref = pypsa.Network()
            extra_net.transformer_types = _ref.transformer_types
            extra_net.line_types        = _ref.line_types
            register_custom_transformer_types(extra_net)
            extra_net.calculate_dependent_values()
            # Apply same customisation steps
            extra_active = get_active_generators(
                extra_net, paths["annual"], year, month_num)
            extra_net = set_dispatchable_availability(
                extra_net, extra_active, availability,
                month_num=month_num, hydro_seasonal=hydro_seasonal)
            # Wind profiles
            month_code = f"{year}{month_num:02d}"
            wind_path  = paths["wind_mon_root"] / str(year) / f"wind_p_pu_{month_code}.csv"
            if wind_path.exists():
                wind_df   = pd.read_csv(str(wind_path), index_col=0, parse_dates=True)
                wind_df.index = pd.to_datetime(wind_df.index)
                extra_net = assign_renewable_profiles(
                    extra_net, wind_df, "wind", wind_mapping)
                extra_net = set_brownfield_expansion_limits(
                    extra_net, "wind", wind_exp_factors)
            # Solar profiles
            solar_year_file  = f"solar_PV_p_pu_{year}.csv"
            solar_alias_file = config["filename"].get("solar", "solar_PV_p_pu.csv")
            solar_path = paths["solar"] / solar_year_file
            if not solar_path.exists():
                solar_path = paths["solar"] / solar_alias_file
            if solar_path.exists():
                solar_df = pd.read_csv(str(solar_path), index_col=0, parse_dates=True)
                solar_df.index = pd.to_datetime(solar_df.index)
                # Remove duplicate timestamps if present
                solar_df = solar_df[~solar_df.index.duplicated(keep="first")]
                extra_net = assign_renewable_profiles(
                    extra_net, solar_df, "solar_pv", solar_mapping)
                extra_net = set_brownfield_expansion_limits(
                    extra_net, "solar_pv", solar_exp_factors)
            extra_net = add_greenfield_generators(extra_net, config, year)
            extra_net = setup_costs(extra_net, paths, config, year)
            extra_net.export_to_csv_folder(str(extra_dir))
            log.info("    Saved to %s", extra_dir)
        except Exception as exc:
            log.error("    Failed %s: %s", extra_label, exc)
            import traceback; traceback.print_exc()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Customise all monthly networks for all configured years.
    """
    import time
    start_time = time.time()

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

    start_year = config["scenario_years"]["start"]
    end_year   = config["scenario_years"]["end"]
    years      = list(range(start_year, end_year + 1))
    scenario   = config.get("scenario", "reference")

    log.info("\n%s", "#" * 60)
    log.info("# NETWORK CUSTOMISATION")
    log.info("# Scenario : %s", scenario)
    log.info("# Years    : %d - %d  (%d years)", start_year, end_year, len(years))
    log.info("# Networks : %d", len(years) * 12)
    log.info("%s", "#" * 60)

    for year in years:
        log.info("\n%s", "#" * 60)
        log.info("# YEAR %d", year)
        log.info("%s", "#" * 60)

        for month_num in range(1, 13):
            try:
                process_monthly_network(month_num, year, config, paths)
            except Exception as exc:
                log.error("Error: %s %d/%02d -- %s",
                          scenario, year, month_num, exc)
                import traceback
                traceback.print_exc()
                continue

    elapsed = time.time() - start_time
    log.info("\n%s", "=" * 60)
    log.info("Customisation complete  (%.1f min)", elapsed / 60)
    log.info("%s", "=" * 60)
    log.info("Next: run nza_cx_opt.py to optimise capacity expansion")


if __name__ == "__main__":
    main()
