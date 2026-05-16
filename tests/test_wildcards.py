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
    """README: /Election:Presidential/*/LK — all presidential election years."""

    def setUp(self):
        # Use EC-01 because the TSV has no 'LK' aggregate row
        self.result = db("/Election:Presidential/*/EC-01")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_keys_are_years(self):
        for key in self.result:
            self.assertRegex(
                key, r"^\d{4}$", f"Expected 4-digit year, got {key!r}"
            )

    def test_contains_known_presidential_years(self):
        result_years = set(self.result.keys())
        self.assertTrue(
            _KNOWN_PRESIDENTIAL_YEARS.issubset(result_years),
            f"Missing years: {_KNOWN_PRESIDENTIAL_YEARS - result_years}",
        )

    def test_each_year_value_is_dict(self):
        for year, data in self.result.items():
            self.assertIsInstance(
                data, dict, f"Value for {year} should be a dict"
            )


class TestWildcardWhenByPDs(unittest.TestCase):
    """All presidential elections broken down by polling division."""

    def setUp(self):
        self.result = db("/Election:Presidential/*/EC-01:PDs")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_contains_2024(self):
        self.assertIn("2024", self.result)

    def test_2024_is_nested_dict(self):
        pd_data = self.result["2024"]
        self.assertIsInstance(pd_data, dict)
        self.assertTrue(pd_data, "Expected polling divisions under 2024")


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
        self.assertIn("populationtotal", self.result["measurements"])


if __name__ == "__main__":
    unittest.main()
