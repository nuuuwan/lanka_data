"""
Tests for empty / missing-data behaviour.

README spec:
    A valid query that matches no data returns an empty result with a warning
    explaining why — not an error.
    Errors are reserved for malformed queries.

Network access required for the data-miss tests.
"""

import unittest

from lanka_data import db
from lanka_data.core.Query import Query


class TestMissingYear(unittest.TestCase):
    """Querying a year for which no data file exists returns {}."""

    def test_presidential_2023_returns_empty(self):
        # README example: /Election:Presidential/2023/LK → {} with warning
        result = db("/Election:Presidential/2023/LK")
        self.assertEqual(result, {})

    def test_population_2024_returns_empty(self):
        # Population census data is only available for 2012
        result = db("/Population/2024/LK")
        self.assertEqual(result, {})


class TestMalformedQuery(unittest.TestCase):
    """Malformed queries (wrong segment count) raise ValueError."""

    def test_too_few_segments(self):
        with self.assertRaises(ValueError):
            db("/Population/2012")

    def test_too_many_segments(self):
        with self.assertRaises(ValueError):
            db("/Population/2012/LK/extra")

    def test_empty_path(self):
        with self.assertRaises(ValueError):
            Query("")


if __name__ == "__main__":
    unittest.main()
