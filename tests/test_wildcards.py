"""
Integration tests for wildcard / catalog queries.

README spec:
    /*/2024/LK                       → measurements available for LK in 2024
    /Election:Presidential/*/LK      → dates of presidential elections in LK
    /Election/*/LK                   → dates of all elections in LK

Network access required.
"""

import unittest

from lanka_data import db

_KNOWN_PRESIDENTIAL_YEARS = {
    "1982",
    "1988",
    "1994",
    "1999",
    "2005",
    "2010",
    "2015",
    "2019",
    "2024",
}


class TestWildcardWhen(unittest.TestCase):
    """README: /Election:Presidential/*/EC-01 — list of election years."""

    def setUp(self):
        self.result = db("/Election:Presidential/*/EC-01")

    def test_has_years_key(self):
        self.assertIn("years", self.result)

    def test_years_is_list(self):
        self.assertIsInstance(self.result["years"], list)

    def test_contains_known_presidential_years(self):
        result_years = set(self.result["years"])
        self.assertTrue(
            _KNOWN_PRESIDENTIAL_YEARS.issubset(result_years),
            f"Missing: {_KNOWN_PRESIDENTIAL_YEARS - result_years}",
        )


class TestWildcardWhere(unittest.TestCase):
    """/Election:Presidential/2024/* → list of entity IDs."""

    def setUp(self):
        self.result = db("/Election:Presidential/2024/*")

    def test_has_entities_key(self):
        self.assertIn("entities", self.result)

    def test_entities_is_list(self):
        self.assertIsInstance(self.result["entities"], list)

    def test_contains_ec01(self):
        self.assertIn("EC-01", self.result["entities"])


class TestWildcardWhenAndWhere(unittest.TestCase):
    """/Election:Presidential/*/* → years list from index metadata."""

    def setUp(self):
        self.result = db("/Election:Presidential/*/*")

    def test_has_years_key(self):
        self.assertIn("years", self.result)

    def test_contains_2024(self):
        self.assertIn("2024", self.result["years"])


class TestWildcardWhat(unittest.TestCase):
    """README: /*/2024/LK — catalog of all measurements for 2024/LK."""

    def setUp(self):
        # Use 2012 where population data actually exists
        self.result = db("/*/2012/LK")

    def test_returns_dict_with_measurements_key(self):
        self.assertIsInstance(self.result, dict)
        self.assertIn("measurements", self.result)

    def test_measurements_is_list(self):
        self.assertIsInstance(self.result["measurements"], list)

    def test_measurements_not_empty(self):
        self.assertTrue(self.result["measurements"])

    def test_measurements_contains_population(self):
        self.assertIn("Population", self.result["measurements"])


if __name__ == "__main__":
    unittest.main()
