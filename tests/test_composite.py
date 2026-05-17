"""
Integration tests for composite measurements — full result and summary.

README spec:
    /Election:Presidential/2024/LK              → Parties + Summary
    /Election:Presidential:Summary/2024/LK      → just summary

Network access required.
"""

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


class TestElectionPresidentialFull(unittest.TestCase):
    """No sub-component → returns both party columns and summary columns."""

    def setUp(self):
        self.result = db("/Election:Presidential/2024/EC-01")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_has_summary_columns(self):
        present = _SUMMARY_COLS & self.result.keys()
        self.assertTrue(present, "Expected at least one summary column")

    def test_has_party_columns(self):
        party_cols = [k for k in self.result if k not in _SUMMARY_COLS]
        self.assertTrue(party_cols, "Expected at least one party column")


class TestElectionPresidentialSummaryOnly(unittest.TestCase):
    """README: /Election:Presidential:Summary/2024/LK → just summary."""

    def setUp(self):
        self.result = db("/Election:Presidential:Summary/2024/EC-01")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_contains_only_summary_cols(self):
        for key in self.result:
            self.assertIn(
                key, _SUMMARY_COLS, f"Unexpected non-summary column: {key!r}"
            )

    def test_has_core_summary_fields(self):
        for field in ("valid", "rejected", "polled", "electors"):
            self.assertIn(field, self.result)


if __name__ == "__main__":
    unittest.main()
