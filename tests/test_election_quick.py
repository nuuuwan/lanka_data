"""
Integration tests: election quick examples from README.md.
Network access required.
"""

import re
import unittest

from lanka_data import db

_SUMMARY_COLS = {
    "valid",
    "rejected",
    "polled",
    "electors",
    "turnout",
    "p_value",
    "seats",
}


class TestElectionPresidentialNational(unittest.TestCase):
    """README: /Election:Presidential/2024/LK —
    full results at national level."""

    def setUp(self):
        self.result = db("/Election:Presidential/2024/LK")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_not_empty(self):
        self.assertTrue(self.result)

    def test_has_summary_fields(self):
        for field in ("valid", "rejected", "polled", "electors"):
            self.assertIn(field, self.result)

    def test_has_party_fields(self):
        party_cols = [k for k in self.result if k not in _SUMMARY_COLS]
        self.assertTrue(party_cols, "Expected at least one party column")


class TestElectionPresidentialSummaryByPDs(unittest.TestCase):
    """
    README: /Election:Presidential:Summary/2024/EC-01:PDs
    Summary per polling division.
    """

    def setUp(self):
        self.result = db("/Election:Presidential:Summary/2024/EC-01:PDs")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_not_empty(self):
        self.assertTrue(self.result)

    def test_keys_are_polling_divisions(self):
        pattern = re.compile(r"^EC-01[A-Z]$")
        for key in self.result:
            self.assertRegex(key, pattern)

    def test_each_pd_has_summary_cols(self):
        for pd_data in self.result.values():
            self.assertIn("valid", pd_data)
            self.assertIn("electors", pd_data)


if __name__ == "__main__":
    unittest.main()
