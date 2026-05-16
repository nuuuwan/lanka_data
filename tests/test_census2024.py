"""Integration tests for Census 2024 data repo."""

import unittest

from lanka_data import db


class TestCensusPopulationGender(unittest.TestCase):
    """Full query: returns entity → value dict."""

    def setUp(self):
        self.result = db("/Gender/2024/LK")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_lk(self):
        self.assertIn("LK", self.result)

    def test_lk_value_is_dict(self):
        self.assertIsInstance(self.result["LK"], dict)

    def test_total_population(self):
        self.assertEqual(self.result["LK"]["total"], 21781800)


class TestCensusHousing(unittest.TestCase):
    """Single-column dataset (Census-only): returns scalar value."""

    def setUp(self):
        self.result = db("/Housing/2024/LK")

    def test_has_lk(self):
        self.assertIn("LK", self.result)

    def test_lk_value_is_int(self):
        self.assertIsInstance(self.result["LK"], int)


class TestCensusWrongYear(unittest.TestCase):
    def test_returns_empty_for_non_census_year(self):
        # Housing is Census-only; 2012 exists in neither repo
        self.assertEqual(db("/Housing/2012/LK"), {})


class TestCensusWildcardWhen(unittest.TestCase):
    """/Housing/*/<where> — Census-only label, only 2024 exists."""

    def setUp(self):
        self.result = db("/Housing/*/LK")

    def test_has_years_key(self):
        self.assertIn("years", self.result)

    def test_includes_2024(self):
        self.assertIn("2024", self.result["years"])

    def test_only_2024(self):
        self.assertEqual(self.result["years"], ["2024"])


class TestCensusOverlapWildcardWhen(unittest.TestCase):
    """/Gender/*/LK — both repos contribute years."""

    def setUp(self):
        self.result = db("/Gender/*/LK")

    def test_has_years_key(self):
        self.assertIn("years", self.result)

    def test_includes_2024(self):
        self.assertIn("2024", self.result["years"])

    def test_includes_gig2_year(self):
        self.assertIn("2012", self.result["years"])


class TestCensusWildcardWhere(unittest.TestCase):
    """/Housing/2024/* → {"entities": [...]}."""

    def setUp(self):
        self.result = db("/Housing/2024/*")

    def test_has_entities_key(self):
        self.assertIn("entities", self.result)

    def test_entities_is_list(self):
        self.assertIsInstance(self.result["entities"], list)

    def test_contains_lk(self):
        self.assertIn("LK", self.result["entities"])


class TestCensusWildcardWhenAndWhere(unittest.TestCase):
    """/Housing/*/* → {"years": ["2024"]} (no TSV needed)."""

    def setUp(self):
        self.result = db("/Housing/*/*")

    def test_has_years_key(self):
        self.assertIn("years", self.result)

    def test_includes_2024(self):
        self.assertIn("2024", self.result["years"])


class TestCensusCatalogMerge(unittest.TestCase):
    """/*/2024/LK catalog includes both GIG2 and Census data."""

    def setUp(self):
        self.result = db("/*/2024/LK")

    def test_has_measurements_key(self):
        self.assertIn("measurements", self.result)

    def test_contains_gig2_measurement(self):
        self.assertIn("Election:Presidential", self.result["measurements"])

    def test_contains_census_only_measurement(self):
        self.assertIn("Housing", self.result["measurements"])

    def test_contains_shared_measurement(self):
        self.assertIn("Gender", self.result["measurements"])

    def test_measurements_sorted(self):
        m = self.result["measurements"]
        self.assertEqual(m, sorted(m))


if __name__ == "__main__":
    unittest.main()
