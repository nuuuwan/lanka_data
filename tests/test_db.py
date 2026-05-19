"""Integration tests for Db — the main query entry point."""

import pytest

from lanka_data import Db

# --- Population (GIG2 2012) ---


def test_population_national():
    r = Db("/LK/Population/2012")
    assert r["TotalPopulation"] == 20_357_776


def test_population_districts():
    r = Db("/LK:Districts/Population/2012")
    assert len(r) == 25
    assert r["LK-11"]["TotalPopulation"] == 2_323_964


# --- Census 2024 ---


def test_ethnicity_keys():
    r = Db("/LK/Ethnicity/2024")
    assert r["TotalPopulation"] == 21_781_800
    for key in (
        "Sinhalese",
        "SriLankanTamil",
        "SriLankanMoor",
        "IndianTamil",
    ):
        assert key in r


def test_gender():
    r = Db("/LK/Gender/2024")
    assert r["TotalPopulation"] == 21_781_800
    assert r["Male"] == 10_512_344
    assert r["Female"] == 11_269_456


def test_religion():
    r = Db("/LK/Religion/2024")
    assert r["TotalPopulation"] == 21_781_800
    assert "Buddhist" in r


def test_what_case_insensitive():
    # measurement name is case-insensitive; region codes are not
    assert Db("/LK/ethnicity/2024") == Db("/LK/Ethnicity/2024")


# --- Elections ---


def test_election_national():
    r = Db("/LK/Election:Presidential/2024")
    for field in ("Valid", "Rejected", "Polled", "Electors", "NPP", "SJB"):
        assert field in r


def test_election_summary_pds():
    r = Db("/EC-01:PDs/Election:Presidential:Summary/2024")
    assert "EC-01A" in r
    for field in ("Valid", "Rejected", "Polled", "Electors"):
        assert field in r["EC-01A"]


def test_election_parties_only():
    full = Db("/LK/Election:Presidential/2024")
    summary = Db("/LK/Election:Presidential:Summary/2024")
    parties = Db("/LK/Election:Presidential:Parties/2024")
    summary_keys = set(summary.keys())
    party_keys = set(parties.keys())
    assert summary_keys & party_keys == set()
    assert summary_keys | party_keys == set(full.keys())


# --- Wildcards ---


def test_wildcard_what():
    r = Db("/LK/*/2024")
    assert "measurements" in r
    assert "Ethnicity" in r["measurements"]
    assert "Election:Presidential" in r["measurements"]


def test_wildcard_when():
    r = Db("/LK/Election:Presidential/*")
    assert "years" in r
    assert "2024" in r["years"]
    assert "2019" in r["years"]


# --- Empty / error handling ---


def test_empty_result_bad_year():
    r = Db("/LK/Election:Presidential/2023")
    assert r == {}


def test_malformed_path_raises():
    with pytest.raises(ValueError):
        Db("/only/two")


# --- how segment ---


def test_how_json_explicit():
    # /path/JSON and /path are equivalent
    assert Db("/LK/Gender/2024/JSON") == Db("/LK/Gender/2024")


def test_how_json_case_insensitive():
    assert Db("/LK/Gender/2024/json") == Db("/LK/Gender/2024/JSON")


def test_how_visual_returns_svg():
    for how in ("Bar", "Pie"):
        result = Db(f"/LK/Ethnicity/2024/{how}")
        assert isinstance(result, str)
        assert result.startswith("<svg")
        assert "<metadata>" in result

    # Map requires a sub-region breakdown
    result = Db("/LK:Districts/Ethnicity/2024/Map")
    assert isinstance(result, str)
    assert result.startswith("<svg")
    assert "<metadata>" in result


def test_how_unknown_raises():
    with pytest.raises(ValueError):
        Db("/LK/Ethnicity/2024/XLSX")
