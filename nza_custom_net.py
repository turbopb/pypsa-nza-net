#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nza_custom_net.py

Customise validated PyPSA base networks into operational references.

Reads SI, NI and combined base networks produced by nza_base_net.py and adds:
    - Generator availability (p_max_pu): scalar for dispatchable plant,
      timeseries for wind and solar.
    - Operating cost data for existing generators.

This script prepares operational 2024/2025 reference networks only. It does
not enable brownfield expansion or add greenfield investment candidates;
future candidate investments are introduced by TEFF stage preparation.

All three monthly network variants are required. Customisation is fail-fast:
an error stops the run rather than silently leaving a partially customised set.

Workflow
--------
    nza_base_net.py  -->  base networks (topology + loads, dormant generators)
    nza_custom_net.py -->  fixed operational reference networks
    TEFF preparation  -->  planning-stage candidates and constraints

Validated base networks remain unchanged. Derived operational networks are
written to ``models/networks/customised/<scenario>/<year>/`` with the same
SI, NI and combined monthly folder names.

Author: Phillippe Bruneau
"""

import sys
import logging
import re
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


def site_from_bus(bus: str) -> str:
    """Remove only a trailing voltage suffix from a model bus name."""
    return re.sub(r"_\d{2,3}$", "", str(bus).strip().upper())


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_renewable_profile_contract(config: Dict) -> None:
    """Require an explicit dimensionless p.u. contract for wind and solar."""
    profiles = config.get("renewable_profiles")
    if not isinstance(profiles, dict):
        raise ValueError(
            "Config must define renewable_profiles contracts for wind and solar_pv"
        )

    model_frequency = str(config.get("frequency", "")).strip()
    for carrier in ("wind", "solar_pv"):
        contract = profiles.get(carrier)
        if not isinstance(contract, dict):
            raise ValueError(f"renewable_profiles.{carrier} is required")
        quantity = str(contract.get("quantity", "")).strip()
        unit = str(contract.get("unit", "")).strip()
        source_interval = str(contract.get("source_interval", "")).strip()
        derivation = str(contract.get("derivation", "")).strip()
        if quantity != "per_unit_availability" or unit != "p.u.":
            raise ValueError(
                f"renewable_profiles.{carrier} must declare "
                "quantity=per_unit_availability and unit=p.u."
            )
        if not source_interval or not derivation:
            raise ValueError(
                f"renewable_profiles.{carrier} requires source_interval and derivation"
            )
        if source_interval != model_frequency:
            raise ValueError(
                f"renewable_profiles.{carrier}.source_interval={source_interval!r} "
                f"does not match model frequency {model_frequency!r}; explicit "
                "time-averaging is required before coarser profiles may be used"
            )


def _is_nz_dst_end_repeat(timestamp: pd.Timestamp) -> bool:
    """True for the repeated 02:00/02:30 slots on NZ DST end day."""
    timestamp = pd.Timestamp(timestamp)
    return (
        timestamp.month == 4
        and timestamp.day <= 7
        and timestamp.weekday() == 6
        and timestamp.hour == 2
        and timestamp.minute in {0, 30}
    )


def _is_nz_dst_start_gap(timestamp: pd.Timestamp) -> bool:
    """True for the skipped 02:00/02:30 slots on NZ DST start day."""
    timestamp = pd.Timestamp(timestamp)
    return (
        timestamp.month == 9
        and timestamp.weekday() == 6
        and (timestamp + pd.Timedelta(days=7)).month != 9
        and timestamp.hour == 2
        and timestamp.minute in {0, 30}
    )


def load_profile_csv(
    path: Path,
    label: str,
    normalise_nz_dst: bool = False,
) -> pd.DataFrame:
    """Load one renewable profile file and reject ambiguous input.

    Solar profiles are produced in New Zealand civil time, whereas the model
    uses a fixed 48 timezone-naive snapshots per day. When explicitly enabled,
    the two repeated autumn PV timestamps are averaged. The corresponding
    spring gap is handled during network-snapshot alignment.
    """
    if not path.exists():
        raise FileNotFoundError(f"{label} profile not found: {path}")

    profile = pd.read_csv(path, index_col=0, parse_dates=True)
    profile.index = pd.to_datetime(profile.index, errors="raise")
    if profile.index.has_duplicates:
        duplicates = profile.index[profile.index.duplicated()].unique()
        duplicate_counts = profile.index.value_counts().loc[duplicates]
        valid_dst_repeat = (
            normalise_nz_dst
            and all(_is_nz_dst_end_repeat(value) for value in duplicates)
            and bool(duplicate_counts.eq(2).all())
        )
        if not valid_dst_repeat:
            raise ValueError(
                f"{label} profile contains duplicate timestamps: "
                + ", ".join(str(value) for value in duplicates[:10])
            )

        profile = profile.apply(pd.to_numeric, errors="raise")
        profile = profile.groupby(level=0, sort=True).mean()
        log.info(
            "    %s profile: averaged %d repeated NZ DST timestamps",
            label,
            len(duplicates),
        )

    profile.columns = [str(column).strip().upper() for column in profile.columns]
    duplicate_columns = sorted(
        set(profile.columns[profile.columns.duplicated(keep=False)])
    )
    if duplicate_columns:
        raise ValueError(
            f"{label} profile contains duplicate site columns after "
            f"normalisation: {duplicate_columns}"
        )
    profile = profile.apply(pd.to_numeric, errors="raise")
    return profile.sort_index()


def resolve_paths(config: Dict) -> Dict:
    """Build absolute Path objects for all data inputs and outputs."""
    root = Path(config["paths"]["root"])
    return {
        "networks":    root / config["paths"]["dirpath_networks"],
        "custom_networks": (
            root / config["paths"]["dirpath_custom_networks"]
        ),
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
    year: int,
) -> List[str]:
    """
    Return list of generator names active in this network period.

    The base-network generator inventory defines existing plant. A generator
    is active when its build year is zero (the explicit unknown/planner-managed
    sentinel used for lumped Huntly) or no later than the network year.

    Observed dispatch is deliberately not used as an availability test: a
    plant that did not generate in one historical month still physically
    exists and must remain available to the operational model.

    Parameters
    ----------
    network : pypsa.Network
    year : int

    Returns
    -------
    list of str
        Generator names (as stored in network.generators.index).
    """
    if "build_year" not in network.generators.columns:
        raise ValueError("generators table has no build_year column")

    build_year = pd.to_numeric(
        network.generators["build_year"], errors="coerce"
    )
    invalid = network.generators.index[build_year.isna()]
    if len(invalid):
        raise ValueError(
            "Generators have invalid build_year values: "
            + ", ".join(str(name) for name in invalid)
        )

    active_mask = build_year.eq(0) | build_year.le(year)
    active_gens = list(network.generators.index[active_mask])
    future_gens = list(network.generators.index[~active_mask])

    log.info("    Existing generators active in %d: %d / %d", year,
             len(active_gens), len(network.generators))
    if future_gens:
        log.info(
            "    Not yet commissioned in %d: %s",
            year,
            ", ".join(str(name) for name in future_gens),
        )
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

    # Base inputs are dormant. Reset explicitly so rerunning this function on
    # an in-memory network cannot retain an earlier availability value.
    network.generators.loc[:, "p_max_pu"] = 0.0

    for gen in generator_list:
        if gen not in network.generators.index:
            log.warning("    Generator '%s' not found in network -- skipped", gen)
            continue

        carrier = network.generators.at[gen, "carrier"]

        if carrier in renewable_carriers:
            continue   # handled by timeseries assignment

        if carrier == "hydro" and hydro_seasonal:
            if month_num not in hydro_seasonal:
                raise ValueError(
                    f"hydro_seasonal_availability has no month {month_num}"
                )
            value = float(hydro_seasonal[month_num])
        elif carrier in availability:
            value = float(availability[carrier])
        else:
            raise ValueError(
                f"No availability entry for carrier {carrier!r} "
                f"(generator {gen})"
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Invalid p_max_pu={value} for generator {gen}; "
                "expected a value in [0, 1]"
            )
        network.generators.at[gen, "p_max_pu"] = value

    return network


# =============================================================================
# RENEWABLE PROFILES
# =============================================================================

def assign_renewable_profiles(
    network: pypsa.Network,
    profiles_df: pd.DataFrame,
    carrier: str,
    profile_mapping: Dict[str, str],
    generator_list: List[str],
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
    generators = [
        gen for gen in generator_list
        if network.generators.at[gen, "carrier"] == carrier
    ]

    for gen in generators:
        bus     = network.generators.at[gen, "bus"]
        site    = site_from_bus(bus)
        profile_site = str(profile_mapping.get(site, site)).strip().upper()

        if profile_site not in profiles_df.columns:
            raise ValueError(
                f"{gen}: no {carrier} profile found; checked "
                f"column {profile_site!r}"
            )

        profile = profiles_df[profile_site].reindex(network.snapshots)

        missing = int(profile.isna().sum())
        if missing:
            missing_timestamps = profile.index[profile.isna()]
            valid_solar_gap = (
                carrier == "solar_pv"
                and all(
                    _is_nz_dst_start_gap(value)
                    for value in missing_timestamps
                )
            )
            if valid_solar_gap:
                profile.loc[missing_timestamps] = 0.0
                log.info(
                    "    %s: inserted zero PV for %d NZ DST spring slots",
                    gen,
                    missing,
                )
            else:
                raise ValueError(
                    f"{gen}: {carrier} profile is missing {missing} of "
                    f"{len(network.snapshots)} network snapshots"
                )

        profile = pd.to_numeric(profile, errors="coerce")
        if profile.isna().any() or not np.isfinite(profile.to_numpy()).all():
            raise ValueError(f"{gen}: {carrier} profile contains non-numeric values")

        lower = float(profile.min())
        upper = float(profile.max())
        if lower < -1e-9 or upper > 1.0 + 1e-9:
            raise ValueError(
                f"{gen}: {carrier} profile range [{lower}, {upper}] "
                "is outside [0, 1]"
            )
        profile = profile.clip(lower=0.0, upper=1.0)

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
# LEGACY EXPANSION HELPERS -- NOT USED BY THE REFERENCE-NETWORK WORKFLOW
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
# GREENFIELD GENERATION (LEGACY HELPER -- NOT INVOKED HERE)
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

def load_cost_data(cost_path: Path) -> Dict:
    """Load technology cost data from YAML file."""
    if not cost_path.exists():
        raise FileNotFoundError(f"Cost file not found: {cost_path}")
    with open(cost_path, "r") as f:
        return yaml.safe_load(f)


def apply_generator_costs(
    network: pypsa.Network,
    costs: Dict,
) -> pypsa.Network:
    """
    Apply operating costs to the fixed existing-generator inventory.

    Existing reference assets are non-extendable, so their capital cost is
    set to zero without an arbitrary age/write-down rule. Candidate investment
    capital costs are applied later by TEFF stage preparation. The explicit
    ``marginal_cost_nzd_per_mwh`` value is copied unchanged from the reviewed
    cost file.

    Parameters
    ----------
    network : pypsa.Network
    costs : dict
        Technology cost data from YAML.
    Returns
    -------
    pypsa.Network
    """
    # Import standard_modality inline to avoid circular dependency
    def _std(modality: str) -> str:
        MODALITY_MAP = {
            "wind": "wind", "solar_pv": "solar_pv", "hydro": "hydro",
            "geothermal": "geothermal", "ocgt": "ocgt", "ccgt": "ccgt",
            "gas": "gas", "coal_gas": "coal_gas", "coal": "coal",
            "diesel": "diesel", "battery": "battery",
            "pumped_hydro": "pumped_hydro", "bioenergy": "bioenergy",
            "off_wind": "wind_offshore",
        }
        return MODALITY_MAP.get(modality.lower(), modality.lower())

    carriers = network.generators["carrier"].unique()

    for carrier in carriers:
        carrier_std = _std(carrier)

        if carrier_std not in costs:
            raise ValueError(
                f"No cost data for generator carrier {carrier!r} "
                f"(expected technology key {carrier_std!r})"
            )

        cost_data = costs[carrier_std]
        if "marginal_cost_nzd_per_mwh" not in cost_data:
            raise ValueError(
                f"Cost entry {carrier_std!r} has no "
                "marginal_cost_nzd_per_mwh"
            )
        marginal_cost = float(cost_data["marginal_cost_nzd_per_mwh"])
        if not np.isfinite(marginal_cost) or marginal_cost < 0.0:
            raise ValueError(
                f"Invalid marginal cost for {carrier_std!r}: "
                f"{marginal_cost}"
            )

        mask = network.generators["carrier"] == carrier

        for gen in network.generators[mask].index:
            network.generators.at[gen, "capital_cost"] = 0.0
            network.generators.at[gen, "marginal_cost"] = marginal_cost

    return network


def setup_costs(
    network: pypsa.Network,
    paths: Dict,
    config: Dict,
) -> pypsa.Network:
    """Load cost YAML and apply to network."""
    cost_path = paths["costs"] / config["filename"]["costs_file"]
    costs     = load_cost_data(cost_path)
    network   = apply_generator_costs(network, costs)
    return network


# =============================================================================
# REFERENCE-NETWORK CONTRACT
# =============================================================================

def validate_reference_network(network: pypsa.Network) -> None:
    """Fail if investment candidates or extendable assets are present."""
    violations = []

    greenfield_generators = [
        str(name) for name in network.generators.index
        if str(name).endswith("_greenfield")
    ]
    greenfield_storage = [
        str(name) for name in network.storage_units.index
        if str(name).endswith("_greenfield")
    ]

    if greenfield_generators:
        violations.append(
            "greenfield generators: " + ", ".join(greenfield_generators)
        )
    if greenfield_storage:
        violations.append(
            "greenfield storage units: " + ", ".join(greenfield_storage)
        )

    extendable_components = [
        ("generators", network.generators, "p_nom_extendable"),
        ("storage units", network.storage_units, "p_nom_extendable"),
        ("links", network.links, "p_nom_extendable"),
        ("lines", network.lines, "s_nom_extendable"),
        ("transformers", network.transformers, "s_nom_extendable"),
    ]

    def _as_bool(value) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    for label, table, attribute in extendable_components:
        if attribute not in table.columns or table.empty:
            continue
        extendable = table.index[table[attribute].map(_as_bool)]
        if len(extendable):
            violations.append(
                f"extendable {label}: "
                + ", ".join(str(name) for name in extendable)
            )

    if violations:
        raise ValueError(
            "Reference-network contract violated -- " + "; ".join(violations)
        )


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def _load_network_for_customisation(network_dir: Path) -> pypsa.Network:
    """
    Load one exported base network and restore PyPSA type libraries.

    Parameters
    ----------
    network_dir : pathlib.Path
        Directory containing an exported PyPSA CSV network.

    Returns
    -------
    pypsa.Network
        Imported network with standard line and transformer type libraries
        restored and dependent electrical values recalculated.
    """
    network = pypsa.Network()
    network.import_from_csv_folder(str(network_dir))

    # Built-in component type libraries are not fully preserved by the CSV
    # export/import cycle. Restore them from a fresh PyPSA Network instance.
    reference = pypsa.Network()
    network.transformer_types = reference.transformer_types
    network.line_types = reference.line_types
    register_custom_transformer_types(network)

    network.calculate_dependent_values()
    network.consistency_check()
    return network


def _customise_network_variant(
    network: pypsa.Network,
    year: int,
    month_num: int,
    config: Dict,
    paths: Dict,
    wind_df: Optional[pd.DataFrame],
    solar_df: Optional[pd.DataFrame],
) -> pypsa.Network:
    """
    Apply the standard customisation sequence to one network variant.

    The same function is used for South Island, North Island and combined NZ
    networks so that all three representations receive identical treatment.

    Parameters
    ----------
    network : pypsa.Network
        Imported base network.
    year : int
        Simulation year.
    month_num : int
        Calendar month number, 1-12.
    config : dict
        Loaded network configuration.
    paths : dict
        Resolved workspace paths.
    wind_df : pandas.DataFrame or None
        Wind p_max_pu profiles for this month.
    solar_df : pandas.DataFrame or None
        Solar p_max_pu profiles.

    Returns
    -------
    pypsa.Network
        Customised network, not yet written to disk.
    """
    availability = config.get("availability", {})
    wind_mapping = config.get("WIND_PROFILE_MAPPING", {})
    solar_mapping = config.get("SOLAR_PROFILE_MAPPING", {})
    hydro_seasonal = config.get("hydro_seasonal_availability", {})
    hydro_seasonal = {int(k): float(v) for k, v in hydro_seasonal.items()}
    expected_months = set(range(1, 13))
    if set(hydro_seasonal) != expected_months:
        raise ValueError(
            "hydro_seasonal_availability must define exactly months 1-12"
        )
    invalid_hydro = {
        month: value
        for month, value in hydro_seasonal.items()
        if not 0.0 <= value <= 1.0
    }
    if invalid_hydro:
        raise ValueError(
            f"Invalid hydro seasonal p_max_pu values: {invalid_hydro}"
        )

    log.info("\n  Step 1: Identifying active generators")
    active_gens = get_active_generators(network, year)

    log.info("\n  Step 2: Setting dispatchable availability")
    network = set_dispatchable_availability(
        network,
        active_gens,
        availability,
        month_num=month_num,
        hydro_seasonal=hydro_seasonal,
    )

    if hydro_seasonal:
        hydro_pu = hydro_seasonal.get(
            month_num, availability.get("hydro", 0.75)
        )
        log.info(
            "    Hydro p_max_pu this month (month %d): %.2f",
            month_num,
            hydro_pu,
        )

    log.info("\n  Step 3: Assigning wind profiles")
    if wind_df is None:
        raise ValueError("Wind profile dataframe was not loaded")
    network = assign_renewable_profiles(
        network, wind_df, "wind", wind_mapping, active_gens
    )

    log.info("\n  Step 4: Assigning solar profiles")
    if solar_df is None:
        raise ValueError("Solar profile dataframe was not loaded")
    network = assign_renewable_profiles(
        network, solar_df, "solar_pv", solar_mapping, active_gens
    )

    log.info("\n  Step 5: Applying generator costs")
    network = setup_costs(network, paths, config)

    log.info("\n  Step 6: Validating reference-network contract")
    validate_reference_network(network)
    network.calculate_dependent_values()
    network.consistency_check()
    return network


def process_monthly_network(
    month_num: int,
    year: int,
    config: Dict,
    paths: Dict,
) -> None:
    """
    Customise all three network representations for one month.

    The required representations are:

        - South Island: ``si_<month>_<year>``
        - North Island: ``ni_<month>_<year>``
        - Combined NZ: ``<month>_<year>``

    All three input folders must exist. The three networks are loaded and fully
    customised before any derived output is written. This prevents a
    customisation error in one representation from silently producing an
    apparently complete, mixed base/customised monthly dataset.

    Parameters
    ----------
    month_num : int
        Month number 1-12.
    year : int
        Simulation year.
    config : dict
        Loaded YAML configuration.
    paths : dict
        Resolved workspace paths.

    Raises
    ------
    FileNotFoundError
        If any required monthly network folder is missing.
    Exception
        Any error encountered while loading or customising a network is
        propagated to ``main`` so the run stops immediately.
    """
    month_name = MONTHS[month_num - 1]
    scenario = config.get("scenario", "reference")
    base_label = f"{month_name}_{year}"
    input_base = paths["networks"] / scenario / str(year)
    output_base = paths["custom_networks"] / scenario / str(year)

    variants = [
        (f"si_{base_label}", "South Island"),
        (f"ni_{base_label}", "North Island"),
        (base_label, "Combined NZ"),
    ]

    missing = [
        str(input_base / label)
        for label, _ in variants
        if not (input_base / label).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Required monthly network folder(s) missing:\n  "
            + "\n  ".join(missing)
        )

    # Load shared renewable-profile inputs once for this month.
    month_code = f"{year}{month_num:02d}"
    wind_path = (
        paths["wind_mon_root"]
        / str(year)
        / f"wind_p_pu_{month_code}.csv"
    )
    wind_df = load_profile_csv(wind_path, "Wind")

    solar_year_file = f"solar_PV_p_pu_{year}.csv"
    solar_alias_file = config["filename"].get(
        "solar", "solar_PV_p_pu.csv"
    )
    solar_path = paths["solar"] / solar_year_file
    if not solar_path.exists():
        solar_path = paths["solar"] / solar_alias_file
        log.info(
            "    Year-specific solar file not found -- using alias: %s",
            solar_alias_file,
        )

    solar_df = load_profile_csv(
        solar_path,
        "Solar",
        normalise_nz_dst=True,
    )

    customised = []

    # Customise all variants in memory first. No network is written until all
    # three variants for the month have completed successfully.
    for label, description in variants:
        input_dir = input_base / label
        output_dir = output_base / label

        log.info("\n%s", "=" * 60)
        log.info("Customising  %s [%s]", label, description)
        log.info("%s", "=" * 60)

        network = _load_network_for_customisation(input_dir)
        network = _customise_network_variant(
            network=network,
            year=year,
            month_num=month_num,
            config=config,
            paths=paths,
            wind_df=wind_df,
            solar_df=solar_df,
        )
        customised.append((label, output_dir, network))

    log.info("\n  Step 7: Saving all three monthly variants")
    for label, output_dir, network in customised:
        output_dir.mkdir(parents=True, exist_ok=True)
        network.export_to_csv_folder(str(output_dir))
        log.info("    Saved %s to %s", label, output_dir)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Customise all SI, NI and combined monthly networks for configured years.

    Workspace discovery follows the same mechanism as ``nza_base_net.py``:
    ``nza_root.py`` is loaded from the shared ``analysis/misc`` directory.

    The run is fail-fast. Any error stops processing immediately rather than
    continuing and leaving an apparently complete but partially customised
    network dataset.
    """
    import time

    start_time = time.time()

    # Locate the shared workspace helper relative to this repository.
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

    config = load_config(config_path)
    validate_renewable_profile_contract(config)
    paths = resolve_paths(config)

    start_year = int(config["scenario_years"]["start"])
    end_year = int(config["scenario_years"]["end"])
    years = list(range(start_year, end_year + 1))
    scenario = config.get("scenario", "reference")

    input_root = paths["networks"] / scenario
    output_root = paths["custom_networks"] / scenario
    if input_root.resolve() == output_root.resolve():
        raise ValueError(
            "Customised-network output resolves to the validated base-network "
            "input directory"
        )

    log.info("\n%s", "#" * 60)
    log.info("# NETWORK CUSTOMISATION")
    log.info("# Scenario : %s", scenario)
    log.info(
        "# Years    : %d - %d  (%d years)",
        start_year,
        end_year,
        len(years),
    )
    log.info("# Networks : %d", len(years) * 12 * 3)
    log.info("# Input    : %s", input_root)
    log.info("# Output   : %s", output_root)
    log.info("%s", "#" * 60)

    for year in years:
        log.info("\n%s", "#" * 60)
        log.info("# YEAR %d", year)
        log.info("%s", "#" * 60)

        for month_num in range(1, 13):
            process_monthly_network(
                month_num=month_num,
                year=year,
                config=config,
                paths=paths,
            )

    elapsed = time.time() - start_time
    log.info("\n%s", "=" * 60)
    log.info("Customisation complete  (%.1f min)", elapsed / 60)
    log.info("Built SI, NI and combined NZ customised networks.")
    log.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
