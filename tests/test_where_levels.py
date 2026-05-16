"""
Unit tests for Where level matching — Provinces, Districts, EDs, PDs.
No network access required.
"""

import unittest

from lanka_data.core.Where import Where


class TestWhereLevelProvinces(unittest.TestCase):
    """README example: LK:Provinces → all 9 provinces."""

    def setUp(self):
        self.where = Where("LK:Provinces")

    def test_matches_provinces(self):
        for code in ("LK-1", "LK-2", "LK-9"):
            self.assertTrue(self.where.matches(code), f"{code} should match")

    def test_does_not_match_districts(self):
        for code in ("LK-11", "LK-21", "LK-91"):
            self.assertFalse(
                self.where.matches(code), f"{code} should not match"
            )

    def test_does_not_match_national(self):
        self.assertFalse(self.where.matches("LK"))


class TestWhereLevelDistricts(unittest.TestCase):
    """
    README examples: LK:Districts (all 25 districts),
    LK-3:Districts (3 districts in Southern Province).
    """

    def test_lk_districts_matches_district_codes(self):
        where = Where("LK:Districts")
        for code in ("LK-11", "LK-21", "LK-31", "LK-45", "LK-91"):
            self.assertTrue(where.matches(code), f"{code} should match")

    def test_lk_districts_does_not_match_province(self):
        self.assertFalse(Where("LK:Districts").matches("LK-1"))

    def test_lk_districts_does_not_match_national(self):
        self.assertFalse(Where("LK:Districts").matches("LK"))

    def test_lk3_districts_matches_southern_districts(self):
        where = Where("LK-3:Districts")
        for code in ("LK-31", "LK-32", "LK-33"):
            self.assertTrue(where.matches(code), f"{code} should match")

    def test_lk3_districts_does_not_match_other_provinces(self):
        where = Where("LK-3:Districts")
        for code in ("LK-11", "LK-21", "LK-41"):
            self.assertFalse(where.matches(code), f"{code} should not match")

    def test_lk3_districts_does_not_match_province_itself(self):
        self.assertFalse(Where("LK-3:Districts").matches("LK-3"))


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
