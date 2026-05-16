"""
Unit tests for Query parsing and wildcard detection.
No network access required.
"""

import unittest

from lanka_data.core.Query import Query


class TestQueryParsing(unittest.TestCase):

    def test_basic_parse(self):
        q = Query("/Population/2012/LK")
        self.assertEqual(q.what_raw, "Population")
        self.assertEqual(q.when_raw, "2012")
        self.assertEqual(q.where_raw, "LK")

    def test_what_with_sub_path(self):
        q = Query("/Population:Ethnicity/2012/LK:Districts")
        self.assertEqual(q.what_parts, ["Population", "Ethnicity"])
        self.assertEqual(q.where_raw, "LK:Districts")

    def test_year_extraction(self):
        self.assertEqual(Query("/Population/2012/LK").year, "2012")
        self.assertEqual(Query("/Population/2024-09/LK").year, "2024")
        self.assertEqual(Query("/Population/2024-09-21/LK").year, "2024")

    def test_wildcard_year_returns_none(self):
        self.assertIsNone(Query("/Election:Presidential/*/LK").year)

    def test_invalid_segment_count_raises(self):
        with self.assertRaises(ValueError):
            Query("/Population/2012")
        with self.assertRaises(ValueError):
            Query("Population/2012/LK/extra")


class TestQueryWildcards(unittest.TestCase):

    def test_wildcard_what(self):
        q = Query("/*/2024/LK")
        self.assertTrue(q.is_wildcard_what)
        self.assertFalse(q.is_wildcard_when)
        self.assertFalse(q.is_wildcard_where)

    def test_wildcard_when(self):
        q = Query("/Election:Presidential/*/LK")
        self.assertFalse(q.is_wildcard_what)
        self.assertTrue(q.is_wildcard_when)
        self.assertFalse(q.is_wildcard_where)

    def test_wildcard_where(self):
        q = Query("/Population/2012/*")
        self.assertFalse(q.is_wildcard_what)
        self.assertFalse(q.is_wildcard_when)
        self.assertTrue(q.is_wildcard_where)

    def test_wildcard_what_has_no_what_parts(self):
        self.assertEqual(Query("/*/2024/LK").what_parts, [])


if __name__ == "__main__":
    unittest.main()
