"""Focused regression tests for annual MWh/MW handling in nza_base_net."""

from pathlib import Path
import sys
import tempfile
import types
import unittest

import pandas as pd

try:
    import pypsa  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("pypsa")
    stub.Network = object
    sys.modules["pypsa"] = stub

from nza_base_net import fixed_interval_hours, load_annual_timeseries, make_snapshots


def write_annual_pair(folder: Path, year: int, values) -> None:
    index = pd.date_range(f"{year}-01-01", periods=len(values), freq="30min")
    frame = pd.DataFrame({"AAA": values}, index=index)
    frame.to_csv(folder / f"{year}_export_sites_all.csv")
    frame.to_csv(folder / f"{year}_import_sites_all.csv")


class TimeseriesUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_half_hour_interval_energy_converts_to_average_mw(self) -> None:
        write_annual_pair(self.folder, 2025, [1.0, 2.0, 3.0, 4.0])
        export, imported = load_annual_timeseries(
            self.folder,
            2025,
            freq_source="30min",
            freq_target="30min",
            input_quantity="interval_energy",
            input_unit="MWh",
        )
        self.assertEqual(export["AAA"].tolist(), [2.0, 4.0, 6.0, 8.0])
        self.assertTrue(imported.equals(export))
        self.assertAlmostEqual(export["AAA"].sum() * 0.5, 10.0)

    def test_interval_energy_resamples_before_power_conversion(self) -> None:
        write_annual_pair(self.folder, 2025, [1.0, 2.0, 3.0, 4.0])
        export, _ = load_annual_timeseries(
            self.folder,
            2025,
            freq_source="30min",
            freq_target="1h",
            input_quantity="interval_energy",
            input_unit="MWh",
        )
        self.assertEqual(export["AAA"].tolist(), [3.0, 7.0])
        self.assertAlmostEqual(export["AAA"].sum(), 10.0)

    def test_average_power_is_time_averaged(self) -> None:
        write_annual_pair(self.folder, 2025, [2.0, 4.0, 6.0, 8.0])
        export, _ = load_annual_timeseries(
            self.folder,
            2025,
            freq_source="30min",
            freq_target="1h",
            input_quantity="average_power",
            input_unit="MW",
        )
        self.assertEqual(export["AAA"].tolist(), [3.0, 7.0])
        self.assertAlmostEqual(export["AAA"].sum(), 10.0)

    def test_quantity_unit_mismatch_fails(self) -> None:
        write_annual_pair(self.folder, 2025, [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "Unsupported annual timeseries"):
            load_annual_timeseries(
                self.folder,
                2025,
                input_quantity="interval_energy",
                input_unit="MW",
            )

    def test_fixed_interval_duration(self) -> None:
        self.assertAlmostEqual(fixed_interval_hours("30min"), 0.5)
        self.assertAlmostEqual(fixed_interval_hours("1h"), 1.0)
        with self.assertRaisesRegex(ValueError, "not a fixed-duration interval"):
            fixed_interval_hours("1MS")

    def test_monthly_snapshot_boundaries_follow_frequency(self) -> None:
        half_hourly = make_snapshots(2025, 1, "30min")
        hourly = make_snapshots(2025, 1, "1h")
        self.assertEqual(len(half_hourly), 31 * 48)
        self.assertEqual(len(hourly), 31 * 24)
        self.assertEqual(half_hourly[-1], pd.Timestamp("2025-01-31 23:30"))
        self.assertEqual(hourly[-1], pd.Timestamp("2025-01-31 23:00"))


if __name__ == "__main__":
    unittest.main()
