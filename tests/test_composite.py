"""
Integration tests for composite measurements.

README spec:
    /Election:Presidential/2024/LK              → Parties + Summary
    /Election:Presidential:Parties/2024/LK      → just parties
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


class TestElectionPresidentialPartiesOnly(unittest.TestCase):
    """README: /Election:Presidential:Parties/2024/LK → just parties."""

    def setUp(self):
        self.result = db("/Election:Presidential:Parties/2024/EC-01")

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_contains_no_summary_cols(self):
        for key in self.result:
            self.assertNotIn(
                key, _SUMMARY_COLS, f"Unexpected summary column: {key!r}"
            )

    def test_not_empty(self):
        self.assertTrue(self.result, "Expected at least one party column")


class TestCompositeSubComponentSumDoesNotOverlap(unittest.TestCase):
    """Parties + Summary cols should together equal the full result cols."""

    def test_parties_and_summary_partition_full_result(self):
        full = set(db("/Election:Presidential/2024/EC-01").keys())
        summary = set(db("/Election:Presidential:Summary/2024/EC-01").keys())
        parties = set(db("/Election:Presidential:Parties/2024/EC-01").keys())
        self.assertEqual(full, summary | parties)
        self.assertFalse(
            summary & parties,
            "Summary and parties columns must not overlap",
        )


if __name__ == "__main__":
    unittest.main()
