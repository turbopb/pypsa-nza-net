"""Focused tests for explicit source-data AC line ratings."""

import math
import unittest

import pandas as pd
import pypsa

from nza_base_net import add_lines


LINE_TYPE = "490-AL1/64-ST1A 220.0"


def test_lines() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "line": "N-61",
                "bus0": "HLY",
                "bus1": "BOB",
                "volts": 220,
                "config": "double",
                "distance": 44.673,
                "s_nom_mva": 1388.66,
            },
            {
                "line": "N-99",
                "bus0": "HLY",
                "bus1": "BOB",
                "volts": 220,
                "config": "single",
                "distance": 10.0,
                "s_nom_mva": "",
            },
        ]
    )


def empty_network() -> pypsa.Network:
    network = pypsa.Network()
    network.add("Bus", "HLY", v_nom=220.0)
    network.add("Bus", "BOB", v_nom=220.0)
    return network


class LineRatingSourceDataTests(unittest.TestCase):
    def generic_rating(self, network: pypsa.Network, parallel: float) -> float:
        i_nom = float(network.line_types.at[LINE_TYPE, "i_nom"])
        return math.sqrt(3.0) * 220.0 * i_nom * parallel

    def test_only_named_line_receives_explicit_rating(self) -> None:
        network = empty_network()
        add_lines(network, test_lines(), {}, config={})
        self.assertAlmostEqual(
            float(network.lines.at["N-61", "s_nom"]),
            1388.66,
        )
        self.assertAlmostEqual(
            float(network.lines.at["N-99", "s_nom"]),
            self.generic_rating(network, 1.0),
        )

    def test_no_calibration_retains_generic_rating(self) -> None:
        network = empty_network()
        lines = test_lines().drop(columns="s_nom_mva")
        add_lines(network, lines, {}, config={})
        self.assertAlmostEqual(
            float(network.lines.at["N-61", "s_nom"]),
            self.generic_rating(network, 2.0),
        )

    def test_nonpositive_source_rating_fails_fast(self) -> None:
        network = empty_network()
        lines = test_lines()
        lines.loc[lines["line"] == "N-61", "s_nom_mva"] = 0.0
        with self.assertRaisesRegex(ValueError, "s_nom_mva must be finite and positive"):
            add_lines(network, lines, {}, config={})

    def test_nonnumeric_source_rating_fails_fast(self) -> None:
        network = empty_network()
        lines = test_lines()
        lines.loc[lines["line"] == "N-61", "s_nom_mva"] = "not-a-rating"
        with self.assertRaisesRegex(ValueError, "s_nom_mva .* is not numeric"):
            add_lines(network, lines, {}, config={})

    def test_configuration_cannot_override_source_rating(self) -> None:
        network = empty_network()
        stale_config = {
            "line_rating_calibration": {
                "ratings_mva": {"N-61": 9999.0},
            }
        }
        add_lines(network, test_lines(), {}, config=stale_config)
        self.assertAlmostEqual(
            float(network.lines.at["N-61", "s_nom"]),
            1388.66,
        )


if __name__ == "__main__":
    unittest.main()
