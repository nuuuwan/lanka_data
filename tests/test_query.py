"""Tests for Query — path parsing and gig2 key resolution."""

import pytest
from lanka_data.core import Query


# --- Parsing ---

def test_basic_parse():
    q = Query("/Population/2012/LK")
    assert q.what_raw == "Population"
    assert q.year == "2012"
    assert q.where_raw == "LK"


def test_what_parts():
    q = Query("/Election:Presidential:Summary/2024/LK")
    assert q.what_parts == ["Election", "Presidential", "Summary"]


def test_wildcard_what():
    q = Query("/*/2024/LK")
    assert q.is_wildcard_what
    assert not q.is_wildcard_when
    assert not q.is_wildcard_where


def test_wildcard_when():
    q = Query("/Ethnicity/*/LK")
    assert q.is_wildcard_when
    assert not q.is_wildcard_what


def test_malformed_raises():
    with pytest.raises(ValueError):
        Query("/only/two")


# --- gig2_key ---

def test_gig2_key_population():
    key, sub = Query("/Population/2012/LK").gig2_key()
    assert key == "populationtotal"
    assert sub is None


def test_gig2_key_election():
    key, sub = Query("/Election:Presidential/2024/LK").gig2_key()
    assert key == "governmentelectionspresidential"
    assert sub is None


def test_gig2_key_summary():
    key, sub = Query("/Election:Presidential:Summary/2024/LK").gig2_key()
    assert key == "governmentelectionspresidential"
    assert sub == "summary"


def test_gig2_key_wildcard_what():
    key, sub = Query("/*/2024/LK").gig2_key()
    assert key is None
    assert sub is None
