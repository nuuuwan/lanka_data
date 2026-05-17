"""
Integration tests for wildcard / catalog queries.

README spec:
    /*/2024/LK                       → measurements available for LK in 2024
    /Election:Presidential/*/LK      → dates of presidential elections in LK
    /Election/*/LK                   → dates of all elections in LK

Network access required.
"""

import unittest

from lanka_data import Db

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
        self.result = Db("/Election:Presidential/*/EC-01")

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
        self.result = Db("/Election:Presidential/2024/*")

    def test_has_entities_key(self):
        self.assertIn("entities", self.result)

    def test_entities_is_list(self):
        self.assertIsInstance(self.result["entities"], list)

    def test_contains_ec01(self):
        self.assertIn("EC-01", self.result["entities"])


class TestWildcardWhenAndWhere(unittest.TestCase):
    """/Election:Presidential/*/* → years list from index metadata."""

    def setUp(self):
        self.result = Db("/Election:Presidential/*/*")

    def test_has_years_key(self):
        self.assertIn("years", self.result)

    def test_contains_2024(self):
        self.assertIn("2024", self.result["years"])


if __name__ == "__main__":
    unittest.main()
