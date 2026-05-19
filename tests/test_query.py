"""Tests for Query — path parsing and gig2 key resolution."""

import pytest

from lanka_data.core import Query

# --- Parsing ---


def test_basic_parse():
    q = Query("/LK/Population/2012")
    assert q.what_raw == "Population"
    assert q.year == "2012"
    assert q.where_raw == "LK"


def test_what_parts():
    q = Query("/LK/Election:Presidential:Summary/2024")
    assert q.what_parts == ["Election", "Presidential", "Summary"]


def test_wildcard_what():
    q = Query("/LK/*/2024")
    assert q.is_wildcard_what
    assert not q.is_wildcard_when
    assert not q.is_wildcard_where


def test_wildcard_when():
    q = Query("/LK/Ethnicity/*")
    assert q.is_wildcard_when
    assert not q.is_wildcard_what


def test_malformed_raises():
    with pytest.raises(ValueError):
        Query("/only/two")


# --- gig2_key ---


def test_gig2_key_population():
    key, sub = Query("/LK/Population/2012").gig2_key()
    assert key == "populationtotal"
    assert sub is None


def test_gig2_key_election():
    key, sub = Query("/LK/Election:Presidential/2024").gig2_key()
    assert key == "governmentelectionspresidential"
    assert sub is None


def test_gig2_key_summary():
    key, sub = Query("/LK/Election:Presidential:Summary/2024").gig2_key()
    assert key == "governmentelectionspresidential"
    assert sub == "summary"


def test_gig2_key_wildcard_what():
    key, sub = Query("/LK/*/2024").gig2_key()
    assert key is None
    assert sub is None


# --- how ---


def test_how_default_json():
    assert Query("/LK/Population/2012").how == "JSON"


def test_how_explicit():
    assert Query("/LK/Ethnicity/2024/Bar").how == "Bar"
    assert Query("/LK/Ethnicity/2024/Pie").how == "Pie"
    assert Query("/LK/Ethnicity/2024/Map").how == "Map"


def test_how_case_insensitive():
    assert Query("/LK/Ethnicity/2024/bar").how == "Bar"
    assert Query("/LK/Ethnicity/2024/JSON").how == "JSON"


def test_how_unknown_raises():
    with pytest.raises(ValueError):
        Query("/LK/Ethnicity/2024/XLSX")
