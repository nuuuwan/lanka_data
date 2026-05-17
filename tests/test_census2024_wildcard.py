"""Integration tests for Census 2024 — wildcard and catalog queries."""

import unittest

from lanka_data import Db


class TestCensusWildcardWhen(unittest.TestCase):
    """/Housing/*/<where> — Census-only label, only 2024 exists."""

    def setUp(self):
        self.result = Db("/Housing/*/LK")

    def test_has_years_key(self):
        self.assertIn("years", self.result)

    def test_includes_2024(self):
        self.assertIn("2024", self.result["years"])

    def test_only_2024(self):
        self.assertEqual(self.result["years"], ["2024"])


class TestCensusOverlapWildcardWhen(unittest.TestCase):
    """/Gender/*/LK — both repos contribute years."""

    def setUp(self):
        self.result = Db("/Gender/*/LK")

    def test_has_years_key(self):
        self.assertIn("years", self.result)

    def test_includes_2024(self):
        self.assertIn("2024", self.result["years"])

    def test_includes_gig2_year(self):
        self.assertIn("2012", self.result["years"])


class TestCensusWildcardWhere(unittest.TestCase):
    """/Housing/2024/* → {"entities": [...]}."""

    def setUp(self):
        self.result = Db("/Housing/2024/*")

    def test_has_entities_key(self):
        self.assertIn("entities", self.result)

    def test_entities_is_list(self):
        self.assertIsInstance(self.result["entities"], list)

    def test_contains_lk(self):
        self.assertIn("LK", self.result["entities"])


class TestCensusWildcardWhenAndWhere(unittest.TestCase):
    """/Housing/*/* → {"years": ["2024"]} (no TSV needed)."""

    def setUp(self):
        self.result = Db("/Housing/*/*")

    def test_has_years_key(self):
        self.assertIn("years", self.result)

    def test_includes_2024(self):
        self.assertIn("2024", self.result["years"])


class TestCensusCatalogMerge(unittest.TestCase):
    """/*/2024/LK catalog includes both GIG2 and Census data."""

    def setUp(self):
        self.result = Db("/*/2024/LK")

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
