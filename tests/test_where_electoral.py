"""
Unit tests for Where level matching — Electoral Districts and Polling Divs.
No network access required.
"""

import unittest

from lanka_data.core.Where import Where


class TestWhereLevelElectoralDistricts(unittest.TestCase):
    """README example: LK:ElectoralDistricts / LK:EDs."""

    def setUp(self):
        self.where = Where("LK:ElectoralDistricts")

    def test_matches_electoral_districts(self):
        for code in ("EC-01", "EC-10", "EC-22"):
            self.assertTrue(self.where.matches(code), f"{code} should match")

    def test_does_not_match_polling_divisions(self):
        self.assertFalse(self.where.matches("EC-01A"))

    def test_does_not_match_admin_districts(self):
        self.assertFalse(self.where.matches("LK-11"))


class TestWhereLevelPollingDivisions(unittest.TestCase):
    """
    README example: EC-01:PDs
    Polling divisions of Colombo electoral district.
    """

    def setUp(self):
        self.where = Where("EC-01:PDs")

    def test_matches_pds_of_colombo(self):
        for code in ("EC-01A", "EC-01B", "EC-01Z"):
            self.assertTrue(self.where.matches(code), f"{code} should match")

    def test_does_not_match_electoral_district_itself(self):
        self.assertFalse(self.where.matches("EC-01"))

    def test_does_not_match_other_district_pds(self):
        self.assertFalse(self.where.matches("EC-02A"))


if __name__ == "__main__":
    unittest.main()
