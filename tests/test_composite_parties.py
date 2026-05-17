"""
Integration tests for composite election sub-components — parties only.

README spec:
    /Election:Presidential:Parties/2024/LK      → just parties
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
