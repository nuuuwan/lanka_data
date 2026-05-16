"""
Unit tests for Where — single-region matching and level alias resolution.
No network access required.
"""

import unittest

from lanka_data.core.Where import Where


class TestWhereSingleRegion(unittest.TestCase):

    def test_matches_exact_code(self):
        self.assertTrue(Where("LK").matches("LK"))

    def test_does_not_match_other_code(self):
        self.assertFalse(Where("LK").matches("LK-11"))

    def test_ec_code(self):
        self.assertTrue(Where("EC-01").matches("EC-01"))
        self.assertFalse(Where("EC-01").matches("EC-02"))

    def test_district_code(self):
        self.assertTrue(Where("LK-11").matches("LK-11"))
        self.assertFalse(Where("LK-11").matches("LK-12"))

    def test_wildcard_matches_any(self):
        where = Where("*")
        self.assertTrue(where.matches("LK"))
        self.assertTrue(where.matches("EC-01A"))


class TestWhereLevelAliases(unittest.TestCase):
    """README: PDs ↔ PollingDivisions, EDs ↔ ElectoralDistricts."""

    def test_pds_alias(self):
        self.assertEqual(Where("EC-01:PDs").level, "PollingDivisions")

    def test_eds_alias(self):
        self.assertEqual(Where("LK:EDs").level, "ElectoralDistricts")

    def test_canonical_pollingdivisions(self):
        self.assertEqual(
            Where("EC-01:PollingDivisions").level, "PollingDivisions"
        )

    def test_canonical_electoraldistricts(self):
        self.assertEqual(
            Where("LK:ElectoralDistricts").level, "ElectoralDistricts"
        )

    def test_case_insensitive_alias(self):
        self.assertEqual(Where("EC-01:pds").level, "PollingDivisions")
        self.assertEqual(Where("LK:districts").level, "Districts")


if __name__ == "__main__":
    unittest.main()
