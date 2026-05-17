"""
Integration tests: population quick examples from README.md.
Network access required.
"""

import unittest

from lanka_data import Db


class TestPopulationNational(unittest.TestCase):
    """README: /Population/2024/LK — Sri Lanka's total population."""

    def setUp(self):
        self.result = Db("/Population/2012/LK")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_not_empty(self):
        self.assertTrue(self.result)

    def test_has_total_population_key(self):
        self.assertIn("total_population", self.result)

    def test_total_population_value(self):
        self.assertEqual(self.result["total_population"], 20_357_776)


class TestPopulationByDistricts(unittest.TestCase):
    """README: /Population/2024/LK:Districts — breakdown by district."""

    def setUp(self):
        self.result = Db("/Population/2012/LK:Districts")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_25_districts(self):
        self.assertEqual(len(self.result), 25)

    def test_colombo_present(self):
        self.assertIn("LK-11", self.result)

    def test_colombo_population(self):
        self.assertEqual(self.result["LK-11"]["total_population"], 2_323_964)

    def test_each_value_is_dict(self):
        for value in self.result.values():
            self.assertIsInstance(value, dict)


class TestPopulationEthnicityByDistricts(unittest.TestCase):
    """
    README: /Population:Ethnicity/2024/LK:Districts
    Ethnic composition per district.
    """

    def setUp(self):
        self.result = Db("/Population:Ethnicity/2012/LK:Districts")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_25_districts(self):
        self.assertEqual(len(self.result), 25)

    def test_colombo_has_ethnicity_keys(self):
        colombo = self.result["LK-11"]
        self.assertIn("sinhalese", colombo)
        self.assertIn("sl_tamil", colombo)

    def test_nested_values_are_numeric(self):
        colombo = self.result["LK-11"]
        for v in colombo.values():
            self.assertIsInstance(v, (int, float))


if __name__ == "__main__":
    unittest.main()
