"""
Integration tests for wildcard-what (catalog) queries.

README spec:
    /*/2024/LK                       → measurements available for LK in 2024

Network access required.
"""

import unittest

from lanka_data import db


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
