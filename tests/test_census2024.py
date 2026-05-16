"""Integration tests for Census 2024 data repo."""

import unittest

from lanka_data import db


class TestCensusPopulationGender(unittest.TestCase):
    """Full query: returns entity → value dict."""

    def setUp(self):
        self.result = db("/Census:Population:Gender/2024/LK")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_lk(self):
        self.assertIn("LK", self.result)

    def test_lk_value_is_dict(self):
        self.assertIsInstance(self.result["LK"], dict)

    def test_total_population(self):
        self.assertEqual(self.result["LK"]["total"], 21781800)


class TestCensusHousing(unittest.TestCase):
    """Single-column dataset: returns scalar value."""

    def setUp(self):
        self.result = db("/Census:Housing/2024/LK")

    def test_has_lk(self):
        self.assertIn("LK", self.result)

    def test_lk_value_is_int(self):
        self.assertIsInstance(self.result["LK"], int)


class TestCensusWrongYear(unittest.TestCase):
    def test_returns_empty_for_non_census_year(self):
        self.assertEqual(db("/Census:Population:Gender/2012/LK"), {})


class TestCensusWildcardWhen(unittest.TestCase):
    """/Census:.../*/<where> → {"years": ["2024"]}."""

    def setUp(self):
        self.result = db("/Census:Population:Gender/*/LK")

    def test_has_years_key(self):
        self.assertIn("years", self.result)

    def test_includes_2024(self):
        self.assertIn("2024", self.result["years"])

    def test_only_2024(self):
        self.assertEqual(self.result["years"], ["2024"])


class TestCensusWildcardWhere(unittest.TestCase):
    """/Census:.../2024/* → {"entities": [...]}."""

    def setUp(self):
        self.result = db("/Census:Population:Gender/2024/*")

    def test_has_entities_key(self):
        self.assertIn("entities", self.result)

    def test_entities_is_list(self):
        self.assertIsInstance(self.result["entities"], list)

    def test_contains_lk(self):
        self.assertIn("LK", self.result["entities"])


class TestCensusWildcardWhenAndWhere(unittest.TestCase):
    """/Census:.../*/* → {"years": ["2024"]} (no TSV needed)."""

    def setUp(self):
        self.result = db("/Census:Population:Gender/*/*")

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

    def test_contains_census_measurement(self):
        self.assertIn(
            "Census:Population:Gender",
            self.result["measurements"],
        )

    def test_measurements_sorted(self):
        m = self.result["measurements"]
        self.assertEqual(m, sorted(m))


if __name__ == "__main__":
    unittest.main()
