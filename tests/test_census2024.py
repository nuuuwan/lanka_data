"""Integration tests for Census 2024 data repo — specific entity queries."""

import unittest

from lanka_data import Db


class TestCensusPopulationGender(unittest.TestCase):
    """Specific entity: returns flat dict (no entity wrapper)."""

    def setUp(self):
        self.result = Db("/LK/Gender/2024")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_male_key(self):
        self.assertIn("Male", self.result)

    def test_has_total_population_key(self):
        self.assertIn("TotalPopulation", self.result)

    def test_total_population(self):
        self.assertEqual(self.result["TotalPopulation"], 21781800)


class TestCensusHousing(unittest.TestCase):
    """Single-column dataset: returns scalar for specific entity."""

    def setUp(self):
        self.result = Db("/LK/Housing/2024")

    def test_returns_int(self):
        self.assertIsInstance(self.result, int)

    def test_positive_value(self):
        self.assertGreater(self.result, 0)


class TestCensusWrongYear(unittest.TestCase):
    def test_returns_empty_for_non_census_year(self):
        # Housing is Census-only; 2012 exists in neither repo
        self.assertEqual(Db("/LK/Housing/2012"), {})


if __name__ == "__main__":
    unittest.main()
