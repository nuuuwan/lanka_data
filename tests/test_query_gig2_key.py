"""
Unit tests for Query.gig2_key() — key derivation logic.
No network access required.
"""

import unittest

from lanka_data.core.Query import Query


class TestQueryGig2Key(unittest.TestCase):

    def _key(self, path):
        return Query(path).gig2_key()

    def test_population_defaults_to_total(self):
        key, sub = self._key("/Population/2012/LK")
        self.assertEqual(key, "populationtotal")
        self.assertIsNone(sub)

    def test_population_ethnicity(self):
        key, sub = self._key("/Population:Ethnicity/2012/LK")
        self.assertEqual(key, "populationethnicity")
        self.assertIsNone(sub)

    def test_election_presidential(self):
        key, sub = self._key("/Election:Presidential/2024/LK")
        self.assertEqual(key, "governmentelectionspresidential")
        self.assertIsNone(sub)

    def test_election_presidential_summary(self):
        key, sub = self._key("/Election:Presidential:Summary/2024/LK")
        self.assertEqual(key, "governmentelectionspresidential")
        self.assertEqual(sub, "summary")

    def test_election_presidential_parties(self):
        key, sub = self._key("/Election:Presidential:Parties/2024/LK")
        self.assertEqual(key, "governmentelectionspresidential")
        self.assertEqual(sub, "parties")

    def test_election_parliamentary(self):
        key, sub = self._key("/Election:Parliamentary/2024/LK")
        # gig2 prefix is 'government-elections' → normalized includes the 's'
        self.assertEqual(key, "governmentelectionsparliamentary")
        self.assertIsNone(sub)

    def test_wildcard_what_returns_none(self):
        key, sub = self._key("/*/2024/LK")
        self.assertIsNone(key)
        self.assertIsNone(sub)

    def test_normalize_strips_separators(self):
        # AgeGroup in query → age_group in filename — both normalize to
        # 'agegroup'
        key, _ = self._key("/Population:AgeGroup/2012/LK")
        self.assertEqual(Query.normalize("population-age_group"), key)


if __name__ == "__main__":
    unittest.main()
