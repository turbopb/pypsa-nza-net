#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Verify operational networks produced by ``nza_custom_net.py``.

The verifier compares every derived SI, NI and combined monthly network with
its validated base input. It confirms that customisation changes operational
generator data only: topology and effective load timeseries must remain
identical, existing assets must remain non-extendable, renewable profiles must
propagate exactly from their source files, hydro availability must follow the
configured monthly mapping, and generator operating costs must match the
reviewed cost file. Every customised AC line must also retain the finite,
positive ``s_nom`` stored in its corresponding validated base network.

PyPSA CSV exports may represent timeseries rows either by their datetime labels
or by positional keys which refer to ``snapshots.csv``. They may also omit an
all-default-zero timeseries column while retaining its static component. The
checks below resolve both representations to the same effective network data
before comparing them.
"""

from __future__ import annotations

import calendar
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


WORKSPACE = Path(
    r"C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace"
)
CONFIG_PATH = WORKSPACE / "config" / "nza_net_config.yaml"

YEARS = (2024, 2025)
MONTHS = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)

EXPECTED_TOPOLOGY = {
    "SI": {"buses": 79, "lines": 95, "generators": 12, "loads": 45},
    "NI": {"buses": 155, "lines": 189, "generators": 38, "loads": 105},
    "combined": {
        "buses": 234,
        "lines": 284,
        "generators": 50,
        "loads": 150,
    },
}

RENEWABLE_CARRIERS = {"wind", "solar_pv", "off_wind"}
TRUE_VALUES = {"true", "1", "yes"}

COMPONENT_SPECS = {
    "buses": ("buses.csv", ("carrier", "v_nom")),
    "lines": (
        "lines.csv",
        ("bus0", "bus1", "type", "length", "num_parallel", "s_nom"),
    ),
    "transformers": (
        "transformers.csv",
        ("bus0", "bus1", "type", "s_nom"),
    ),
    "links": (
        "links.csv",
        ("bus0", "bus1", "carrier", "p_nom", "p_min_pu", "efficiency"),
    ),
    "generators": (
        "generators.csv",
        ("bus", "carrier", "p_nom", "build_year"),
    ),
    "loads": ("loads.csv", ("bus", "p_set")),
}

NUMERIC_FIELDS = {
    "v_nom", "length", "num_parallel", "s_nom", "p_nom", "p_min_pu",
    "efficiency", "build_year", "p_set",
}

CARRIER_COST_KEYS = {
    "wind": "wind",
    "off_wind": "wind_offshore",
    "solar_pv": "solar_pv",
    "hydro": "hydro",
    "geothermal": "geothermal",
    "ocgt": "ocgt",
    "ccgt": "ccgt",
    "gas": "gas",
    "coal_gas": "coal_gas",
    "coal": "coal",
    "diesel": "diesel",
    "battery": "battery",
    "pumped_hydro": "pumped_hydro",
    "bioenergy": "bioenergy",
}

failures = []
warnings = []
profile_cache: Dict[Tuple[str, int, int], pd.DataFrame] = {}


def fail(message: str) -> None:
    failures.append(message)
    print(f"FAIL: {message}")


def warn(message: str) -> None:
    warnings.append(message)
    print(f"WARN: {message}")


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def fixed_interval_hours(frequency: str) -> float:
    """Return hours represented by one fixed pandas interval."""
    offset = pd.tseries.frequencies.to_offset(frequency)
    hours = offset.nanos / 3_600_000_000_000
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError(f"Invalid fixed frequency {frequency!r}")
    return float(hours)


def validate_data_contracts(config: Dict) -> float:
    """Validate power/energy and renewable-profile unit declarations."""
    frequency = str(config.get("frequency", "")).strip()
    snapshot_hours = fixed_interval_hours(frequency)

    annual = config.get("annual_timeseries")
    if not isinstance(annual, dict):
        raise ValueError("Config is missing annual_timeseries")
    annual_pair = (
        str(annual.get("input_quantity", "")).strip().lower(),
        str(annual.get("input_unit", "")).strip(),
    )
    if annual_pair not in {
        ("interval_energy", "MWh"),
        ("average_power", "MW"),
    }:
        raise ValueError("Invalid annual_timeseries quantity/unit declaration")

    renewable = config.get("renewable_profiles")
    if not isinstance(renewable, dict):
        raise ValueError("Config is missing renewable_profiles")
    for carrier in ("wind", "solar_pv"):
        contract = renewable.get(carrier)
        if not isinstance(contract, dict):
            raise ValueError(f"renewable_profiles.{carrier} is required")
        if (
            str(contract.get("quantity", "")).strip()
            != "per_unit_availability"
            or str(contract.get("unit", "")).strip() != "p.u."
        ):
            raise ValueError(
                f"renewable_profiles.{carrier} must be per_unit_availability/p.u."
            )
        if str(contract.get("source_interval", "")).strip() != frequency:
            raise ValueError(
                f"renewable_profiles.{carrier} interval differs from model frequency"
            )
    return snapshot_hours


def read_component(folder: Path, filename: str) -> pd.DataFrame:
    path = folder / filename
    if not path.exists():
        return pd.DataFrame()
    dataframe = pd.read_csv(path, keep_default_na=False)
    if "name" not in dataframe.columns:
        raise ValueError(f"{path}: missing name column")
    dataframe["name"] = dataframe["name"].astype(str).str.strip()
    if dataframe["name"].eq("").any():
        raise ValueError(f"{path}: contains blank component names")
    if dataframe["name"].duplicated().any():
        duplicates = sorted(
            dataframe.loc[dataframe["name"].duplicated(False), "name"].unique()
        )
        raise ValueError(f"{path}: duplicate component names {duplicates}")
    return dataframe.set_index("name")


def read_network_snapshots(folder: Path) -> Tuple[pd.DatetimeIndex, pd.Index]:
    """Return canonical datetimes and their serialized PyPSA row keys."""
    path = folder / "snapshots.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    dataframe = pd.read_csv(path, index_col=0, keep_default_na=False)
    if "snapshot" not in dataframe.columns:
        raise ValueError(f"{path}: missing snapshot column")

    row_keys = pd.Index(dataframe.index.astype(str))
    snapshots = pd.DatetimeIndex(
        pd.to_datetime(dataframe["snapshot"], errors="raise")
    )
    if row_keys.has_duplicates:
        raise ValueError(f"{path}: duplicate serialized snapshot keys")
    if snapshots.has_duplicates:
        raise ValueError(f"{path}: duplicate timestamps")
    return snapshots, row_keys


def check_snapshot_weightings(folder: Path, expected_hours: float) -> None:
    """Require PyPSA energy/objective weights to equal snapshot duration."""
    path = folder / "snapshots.csv"
    dataframe = pd.read_csv(path, index_col=0, keep_default_na=False)
    for column in ("objective", "stores", "generators"):
        if column not in dataframe.columns:
            fail(f"{path}: missing {column} weighting")
            continue
        values = pd.to_numeric(dataframe[column], errors="coerce")
        if values.isna().any() or not np.allclose(
            values.to_numpy(), expected_hours, rtol=0.0, atol=1e-12
        ):
            observed = sorted(values.dropna().unique().tolist())
            fail(f"{path}: {column} weights {observed}; expected {expected_hours} h")


def read_timeseries(path: Path) -> pd.DataFrame:
    """Read a network timeseries and resolve its keys via snapshots.csv."""
    if not path.exists():
        raise FileNotFoundError(path)

    dataframe = pd.read_csv(path, index_col=0)
    raw_keys = pd.Index(dataframe.index.astype(str))
    if raw_keys.has_duplicates:
        raise ValueError(f"{path}: duplicate timeseries row keys")

    snapshots, serialized_keys = read_network_snapshots(path.parent)
    if len(dataframe) != len(snapshots):
        raise ValueError(
            f"{path}: {len(dataframe)} rows for {len(snapshots)} snapshots"
        )

    if raw_keys.equals(serialized_keys):
        dataframe.index = snapshots
    else:
        try:
            datetime_keys = pd.DatetimeIndex(
                pd.to_datetime(dataframe.index, errors="raise")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: row keys match neither snapshots.csv keys nor "
                "datetime snapshots"
            ) from exc
        if not datetime_keys.equals(snapshots):
            raise ValueError(
                f"{path}: datetime row keys do not match snapshots.csv"
            )
        dataframe.index = snapshots

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    if dataframe.columns.duplicated().any():
        raise ValueError(f"{path}: duplicate timeseries columns")
    return dataframe


def site_from_bus(bus: str) -> str:
    return re.sub(r"_\d{2,3}$", "", str(bus).strip().upper())


def is_nz_dst_end_repeat(timestamp: pd.Timestamp) -> bool:
    timestamp = pd.Timestamp(timestamp)
    return (
        timestamp.month == 4
        and timestamp.day <= 7
        and timestamp.weekday() == 6
        and timestamp.hour == 2
        and timestamp.minute in {0, 30}
    )


def is_nz_dst_start_gap(timestamp: pd.Timestamp) -> bool:
    timestamp = pd.Timestamp(timestamp)
    return (
        timestamp.month == 9
        and timestamp.weekday() == 6
        and (timestamp + pd.Timedelta(days=7)).month != 9
        and timestamp.hour == 2
        and timestamp.minute in {0, 30}
    )


def read_source_profile(path: Path, carrier: str) -> pd.DataFrame:
    """Read a source profile using the customiser's strict DST rule."""
    if not path.exists():
        raise FileNotFoundError(path)
    profile = pd.read_csv(path, index_col=0, parse_dates=True)
    profile.index = pd.to_datetime(profile.index, errors="raise")

    if profile.index.has_duplicates:
        duplicates = profile.index[profile.index.duplicated()].unique()
        counts = profile.index.value_counts().loc[duplicates]
        valid = (
            carrier == "solar_pv"
            and all(is_nz_dst_end_repeat(value) for value in duplicates)
            and bool(counts.eq(2).all())
        )
        if not valid:
            raise ValueError(f"{path}: invalid duplicate timestamps")
        profile = profile.apply(pd.to_numeric, errors="raise")
        profile = profile.groupby(level=0, sort=True).mean()

    profile.columns = [str(column).strip().upper() for column in profile.columns]
    if profile.columns.duplicated().any():
        raise ValueError(f"{path}: duplicate normalised site columns")
    return profile.apply(pd.to_numeric, errors="raise").sort_index()


def compare_value(
    folder: Path,
    component: str,
    name: str,
    field: str,
    base_value,
    custom_value,
) -> None:
    if field in NUMERIC_FIELDS:
        try:
            equal = math.isclose(
                float(base_value), float(custom_value), rel_tol=0.0, abs_tol=1e-8
            )
        except (TypeError, ValueError):
            equal = False
    else:
        equal = str(base_value).strip() == str(custom_value).strip()
    if not equal:
        fail(
            f"{folder}: {component} {name} changed {field} from "
            f"{base_value!r} to {custom_value!r}"
        )


def compare_topology(base_folder: Path, custom_folder: Path) -> Dict[str, pd.DataFrame]:
    custom_components = {}
    for component, (filename, fields) in COMPONENT_SPECS.items():
        base = read_component(base_folder, filename)
        custom = read_component(custom_folder, filename)
        custom_components[component] = custom

        base_names = set(base.index)
        custom_names = set(custom.index)
        missing = sorted(base_names - custom_names)
        unexpected = sorted(custom_names - base_names)
        if missing:
            fail(f"{custom_folder}: missing {component} {missing}")
        if unexpected:
            fail(f"{custom_folder}: unexpected {component} {unexpected}")

        for field in fields:
            if field not in base.columns and field not in custom.columns:
                continue
            if field not in base.columns or field not in custom.columns:
                fail(
                    f"{custom_folder}: {component} field {field!r} is not "
                    "present in both base and customised data"
                )
                continue
            for name in sorted(base_names & custom_names):
                compare_value(
                    custom_folder,
                    component,
                    name,
                    field,
                    base.at[name, field],
                    custom.at[name, field],
                )
    return custom_components


def check_positive_ac_line_ratings(folder: Path, lines: pd.DataFrame) -> None:
    """Require a usable optimisation limit on every customised AC line."""
    if "s_nom" not in lines.columns:
        fail(f"{folder}: lines.csv missing s_nom")
        return

    ratings = pd.to_numeric(lines["s_nom"], errors="coerce")
    valid = ratings.notna() & np.isfinite(ratings) & ratings.gt(0.0)

    if (~valid).any():
        invalid_names = sorted(str(name) for name in lines.index[~valid])
        fail(
            f"{folder}: AC lines have missing, non-numeric, non-finite, "
            f"or non-positive stored s_nom values {invalid_names}"
        )


def compare_load_timeseries(base_folder: Path, custom_folder: Path) -> pd.DatetimeIndex:
    base = read_timeseries(base_folder / "loads-p_set.csv")
    custom = read_timeseries(custom_folder / "loads-p_set.csv")

    if not base.index.equals(custom.index):
        fail(f"{custom_folder}: load snapshot index changed")
        return custom.index

    missing = sorted(set(base.columns) - set(custom.columns))
    unexpected = sorted(set(custom.columns) - set(base.columns))
    if unexpected:
        fail(
            f"{custom_folder}: unexpected load-timeseries columns "
            f"{unexpected}"
        )
        return custom.index

    # PyPSA omits all-default-zero timeseries columns on re-export. This is
    # semantically unchanged only when the static Load component is retained
    # (checked by compare_topology) and the complete base series is zero.
    invalid_missing = []
    for column in missing:
        values = pd.to_numeric(base[column], errors="coerce")
        if values.isna().any() or not np.allclose(
            values.to_numpy(), 0.0, rtol=0.0, atol=1e-12
        ):
            invalid_missing.append(column)
        else:
            custom[column] = 0.0
    if invalid_missing:
        fail(
            f"{custom_folder}: missing non-zero load-timeseries columns "
            f"{invalid_missing}"
        )
        return custom.index

    columns = sorted(base.columns)
    base_values = base[columns].apply(pd.to_numeric, errors="coerce")
    custom_values = custom[columns].apply(pd.to_numeric, errors="coerce")
    if base_values.isna().any().any() or custom_values.isna().any().any():
        fail(f"{custom_folder}: load timeseries contains non-numeric values")
    elif not np.allclose(
        base_values.to_numpy(), custom_values.to_numpy(), rtol=0.0, atol=1e-9
    ):
        fail(f"{custom_folder}: load timeseries changed during customisation")
    return custom.index


def check_non_extendable(folder: Path, components: Dict[str, pd.DataFrame]) -> None:
    checks = (
        ("generators", "p_nom_extendable"),
        ("lines", "s_nom_extendable"),
        ("transformers", "s_nom_extendable"),
        ("links", "p_nom_extendable"),
    )
    for component, attribute in checks:
        table = components[component]
        if table.empty or attribute not in table.columns:
            continue
        mask = (
            table[attribute].astype(str).str.strip().str.lower().isin(TRUE_VALUES)
        )
        if mask.any():
            fail(
                f"{folder}: extendable {component} "
                f"{sorted(str(name) for name in table.index[mask])}"
            )


def source_profile_path(
    config: Dict, carrier: str, year: int, month_number: int
) -> Path:
    root = Path(config["paths"]["root"])
    renewable_root = root / config["paths"]["dirpath_renewables"]
    if carrier == "wind":
        return (
            renewable_root
            / "wind"
            / "wind_p_pu_monthly"
            / str(year)
            / f"wind_p_pu_{year}{month_number:02d}.csv"
        )
    if carrier == "solar_pv":
        solar_root = renewable_root / config["paths"]["dirpath_solar_pu"]
        year_path = solar_root / f"solar_PV_p_pu_{year}.csv"
        if year_path.exists():
            return year_path
        return solar_root / config["filename"].get("solar", "solar_PV_p_pu.csv")
    raise ValueError(f"Unsupported renewable carrier {carrier!r}")


def get_source_profile(
    config: Dict, carrier: str, year: int, month_number: int
) -> pd.DataFrame:
    key = (carrier, year, month_number if carrier == "wind" else 0)
    if key in profile_cache:
        return profile_cache[key]

    path = source_profile_path(config, carrier, year, month_number)
    profile = read_source_profile(path, carrier)
    profile_cache[key] = profile
    return profile


def expected_static_availability(
    carrier: str,
    month_number: int,
    availability: Dict[str, float],
    hydro_seasonal: Dict[int, float],
) -> float:
    if carrier == "hydro":
        return float(hydro_seasonal[month_number])
    if carrier not in availability:
        raise KeyError(f"No availability configured for carrier {carrier!r}")
    return float(availability[carrier])


def check_generators(
    folder: Path,
    generators: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    config: Dict,
    costs: Dict,
    year: int,
    month_number: int,
) -> None:
    required = {"bus", "carrier", "p_nom", "build_year", "marginal_cost"}
    missing = sorted(required - set(generators.columns))
    if missing:
        fail(f"{folder}: generators.csv missing {missing}")
        return

    p_max = (
        pd.to_numeric(generators["p_max_pu"], errors="coerce")
        if "p_max_pu" in generators.columns
        else pd.Series(1.0, index=generators.index)
    )
    build_year = pd.to_numeric(generators["build_year"], errors="coerce")
    availability = {
        str(key): float(value)
        for key, value in config.get("availability", {}).items()
    }
    hydro_seasonal = {
        int(key): float(value)
        for key, value in config.get("hydro_seasonal_availability", {}).items()
    }

    active = build_year.eq(0) | build_year.le(year)
    renewable_names = {
        name
        for name in generators.index[active]
        if str(generators.at[name, "carrier"]) in RENEWABLE_CARRIERS
    }

    profile_path = folder / "generators-p_max_pu.csv"
    if renewable_names:
        try:
            profiles = read_timeseries(profile_path)
        except Exception as exc:
            fail(f"{folder}: cannot read renewable profiles: {exc}")
            profiles = pd.DataFrame(index=snapshots)
    else:
        profiles = pd.DataFrame(index=snapshots)

    if not profiles.index.equals(snapshots):
        fail(f"{folder}: generator-profile snapshot index does not match loads")

    profile_names = set(profiles.columns)
    missing_profiles = sorted(renewable_names - profile_names)
    unexpected_profiles = sorted(profile_names - renewable_names)
    if missing_profiles:
        fail(f"{folder}: missing renewable generator profiles {missing_profiles}")
    if unexpected_profiles:
        fail(f"{folder}: unexpected generator p_max_pu profiles {unexpected_profiles}")

    wind_mapping = {
        str(key).strip().upper(): str(value).strip().upper()
        for key, value in config.get("WIND_PROFILE_MAPPING", {}).items()
    }
    solar_mapping = {
        str(key).strip().upper(): str(value).strip().upper()
        for key, value in config.get("SOLAR_PROFILE_MAPPING", {}).items()
    }

    for name, row in generators.iterrows():
        carrier = str(row["carrier"]).strip()
        is_active = bool(active.at[name])

        if not is_active:
            if not math.isclose(float(p_max.at[name]), 0.0, abs_tol=1e-9):
                fail(f"{folder}: future generator {name} is not dormant")
        elif carrier not in RENEWABLE_CARRIERS:
            try:
                expected = expected_static_availability(
                    carrier, month_number, availability, hydro_seasonal
                )
            except Exception as exc:
                fail(f"{folder}: cannot determine availability for {name}: {exc}")
            else:
                if not math.isclose(
                    float(p_max.at[name]), expected, rel_tol=0.0, abs_tol=1e-9
                ):
                    fail(
                        f"{folder}: {name} p_max_pu={p_max.at[name]}; "
                        f"expected {expected}"
                    )

        cost_key = CARRIER_COST_KEYS.get(carrier, carrier)
        if cost_key not in costs:
            fail(f"{folder}: no cost entry {cost_key!r} for {name}")
        else:
            expected_cost = float(costs[cost_key]["marginal_cost_nzd_per_mwh"])
            actual_cost = float(row["marginal_cost"])
            if not math.isclose(
                actual_cost, expected_cost, rel_tol=0.0, abs_tol=1e-8
            ):
                fail(
                    f"{folder}: {name} marginal_cost={actual_cost}; "
                    f"expected {expected_cost} from {cost_key}"
                )

        capital_cost = float(row.get("capital_cost", 0.0) or 0.0)
        if not math.isclose(capital_cost, 0.0, abs_tol=1e-9):
            fail(f"{folder}: fixed existing generator {name} has capital_cost={capital_cost}")

    for name in sorted(renewable_names & profile_names):
        values = pd.to_numeric(profiles[name], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            fail(f"{folder}: {name} profile contains invalid values")
            continue
        if float(values.min()) < -1e-9 or float(values.max()) > 1.0 + 1e-9:
            fail(f"{folder}: {name} profile lies outside [0, 1]")
            continue

        carrier = str(generators.at[name, "carrier"]).strip()
        if carrier not in {"wind", "solar_pv"}:
            continue
        source = get_source_profile(config, carrier, year, month_number)
        site = site_from_bus(generators.at[name, "bus"])
        mapping = wind_mapping if carrier == "wind" else solar_mapping
        source_site = mapping.get(site, site)
        if source_site not in source.columns:
            fail(f"{folder}: source profile has no column {source_site!r} for {name}")
            continue
        expected = pd.to_numeric(
            source[source_site].reindex(snapshots), errors="coerce"
        )
        if expected.isna().any():
            missing_timestamps = expected.index[expected.isna()]
            valid_solar_gap = (
                carrier == "solar_pv"
                and all(
                    is_nz_dst_start_gap(value)
                    for value in missing_timestamps
                )
            )
            if valid_solar_gap:
                expected.loc[missing_timestamps] = 0.0
            else:
                fail(f"{folder}: source profile is incomplete for {name}")
                continue
        if not np.allclose(
            values.to_numpy(),
            expected.clip(lower=0.0, upper=1.0).to_numpy(),
            rtol=0.0,
            atol=1e-9,
        ):
            fail(f"{folder}: {name} profile differs from source {source_site}")


def check_folder(
    base_folder: Path,
    custom_folder: Path,
    label: str,
    year: int,
    month_number: int,
    config: Dict,
    costs: Dict,
    snapshot_hours: float,
) -> Optional[Dict[str, int]]:
    if not base_folder.is_dir():
        fail(f"Missing base network folder: {base_folder}")
        return None
    if not custom_folder.is_dir():
        fail(f"Missing customised network folder: {custom_folder}")
        return None

    try:
        components = compare_topology(base_folder, custom_folder)
        check_positive_ac_line_ratings(custom_folder, components["lines"])
        snapshots = compare_load_timeseries(base_folder, custom_folder)
        check_snapshot_weightings(base_folder, snapshot_hours)
        check_snapshot_weightings(custom_folder, snapshot_hours)
        check_non_extendable(custom_folder, components)
        check_generators(
            custom_folder,
            components["generators"],
            snapshots,
            config,
            costs,
            year,
            month_number,
        )
    except Exception as exc:
        fail(f"{custom_folder}: verification could not continue: {exc}")
        return None

    counts = {component: len(table) for component, table in components.items()}
    expected = EXPECTED_TOPOLOGY[label]
    for component in ("buses", "lines", "generators", "loads"):
        if counts[component] != expected[component]:
            fail(
                f"{custom_folder}: {component}={counts[component]}; "
                f"expected {expected[component]}"
            )
    return counts


def main() -> None:
    print("=" * 72)
    print("PYPSA-NZA CUSTOMISED NETWORK VERIFICATION")
    print("=" * 72)

    try:
        config = load_yaml(CONFIG_PATH)
        snapshot_hours = validate_data_contracts(config)
        root = Path(config["paths"]["root"])
        scenario = config.get("scenario", "reference")
        base_root = root / config["paths"]["dirpath_networks"] / scenario
        custom_root = (
            root
            / config["paths"]["dirpath_custom_networks"]
            / scenario
        )
        cost_path = (
            root
            / config["paths"]["dirpath_costs"]
            / config["filename"]["costs_file"]
        )
        costs = load_yaml(cost_path)
    except Exception as exc:
        fail(f"Cannot load configuration or costs: {exc}")
        print(f"Failures                : {len(failures)}")
        sys.exit(1)

    hydro_months = {
        int(key) for key in config.get("hydro_seasonal_availability", {})
    }
    if hydro_months != set(range(1, 13)):
        fail("hydro_seasonal_availability must define exactly months 1-12")

    folders_checked = 0
    reference_counts = {}

    for year in YEARS:
        for month_number, month in enumerate(MONTHS, start=1):
            for prefix, label in (("si_", "SI"), ("ni_", "NI"), ("", "combined")):
                folder_name = f"{prefix}{month}_{year}"
                folder_failures_before = len(failures)
                counts = check_folder(
                    base_root / str(year) / folder_name,
                    custom_root / str(year) / folder_name,
                    label,
                    year,
                    month_number,
                    config,
                    costs,
                    snapshot_hours,
                )
                if counts is None:
                    continue
                folders_checked += 1
                if label not in reference_counts:
                    reference_counts[label] = counts
                status = (
                    "OK  "
                    if len(failures) == folder_failures_before
                    else "FAIL"
                )
                print(f"{status} {year} {month.upper():3s} {label:8s}")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Network folders checked : {folders_checked}")
    print(f"Warnings                : {len(warnings)}")
    print(f"Failures                : {len(failures)}")
    for label, counts in reference_counts.items():
        print(
            f"{label:8s} topology       : {counts['buses']} buses, "
            f"{counts['lines']} lines, {counts['transformers']} transformers, "
            f"{counts['links']} links, {counts['generators']} generators, "
            f"{counts['loads']} loads"
        )
    print(f"Snapshot duration       : {snapshot_hours:.6g} h")
    print("Renewable profiles      : dimensionless p.u.; source values unchanged")

    if failures:
        print("\nFAILED CHECKS:")
        for message in failures:
            print(f"  - {message}")
        sys.exit(1)

    print("\nALL CUSTOMISED-NETWORK CHECKS PASSED")


if __name__ == "__main__":
    main()
