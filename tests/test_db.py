"""Integration tests for Db — the main query entry point."""

import pytest

from lanka_data import Db

# --- Population (GIG2 2012) ---


def test_population_national():
    r = Db("/Population/2012/LK")
    assert r["TotalPopulation"] == 20_357_776


def test_population_districts():
    r = Db("/Population/2012/LK:Districts")
    assert len(r) == 25
    assert r["LK-11"]["TotalPopulation"] == 2_323_964


# --- Census 2024 ---


def test_ethnicity_keys():
    r = Db("/Ethnicity/2024/LK")
    assert r["TotalPopulation"] == 21_781_800
    for key in (
        "Sinhalese",
        "SriLankanTamil",
        "SriLankanMoor",
        "IndianTamil",
    ):
        assert key in r


def test_gender():
    r = Db("/Gender/2024/LK")
    assert r["TotalPopulation"] == 21_781_800
    assert r["Male"] == 10_512_344
    assert r["Female"] == 11_269_456


def test_religion():
    r = Db("/Religion/2024/LK")
    assert r["TotalPopulation"] == 21_781_800
    assert "Buddhist" in r


def test_what_case_insensitive():
    # measurement name is case-insensitive; region codes are not
    assert Db("/ethnicity/2024/LK") == Db("/Ethnicity/2024/LK")


# --- Elections ---


def test_election_national():
    r = Db("/Election:Presidential/2024/LK")
    for field in ("Valid", "Rejected", "Polled", "Electors", "NPP", "SJB"):
        assert field in r


def test_election_summary_pds():
    r = Db("/Election:Presidential:Summary/2024/EC-01:PDs")
    assert "EC-01A" in r
    for field in ("Valid", "Rejected", "Polled", "Electors"):
        assert field in r["EC-01A"]


def test_election_parties_only():
    full = Db("/Election:Presidential/2024/LK")
    summary = Db("/Election:Presidential:Summary/2024/LK")
    parties = Db("/Election:Presidential:Parties/2024/LK")
    summary_keys = set(summary.keys())
    party_keys = set(parties.keys())
    assert summary_keys & party_keys == set()
    assert summary_keys | party_keys == set(full.keys())


# --- Wildcards ---


def test_wildcard_what():
    r = Db("/*/2024/LK")
    assert "measurements" in r
    assert "Ethnicity" in r["measurements"]
    assert "Election:Presidential" in r["measurements"]


def test_wildcard_when():
    r = Db("/Election:Presidential/*/LK")
    assert "years" in r
    assert "2024" in r["years"]
    assert "2019" in r["years"]


# --- Empty / error handling ---


def test_empty_result_bad_year():
    r = Db("/Election:Presidential/2023/LK")
    assert r == {}


def test_malformed_path_raises():
    with pytest.raises(ValueError):
        Db("/only/two")


# --- how segment ---


def test_how_json_explicit():
    # /path/JSON and /path are equivalent
    assert Db("/Gender/2024/LK/JSON") == Db("/Gender/2024/LK")


def test_how_json_case_insensitive():
    assert Db("/Gender/2024/LK/json") == Db("/Gender/2024/LK/JSON")


def test_how_visual_returns_svg():
    for how in ("Bar", "Pie"):
        result = Db(f"/Ethnicity/2024/LK/{how}")
        assert isinstance(result, str)
        assert result.startswith("<svg")
        assert "<metadata>" in result

    # Map requires a sub-region breakdown
    result = Db("/Ethnicity/2024/LK:Districts/Map")
    assert isinstance(result, str)
    assert result.startswith("<svg")
    assert "<metadata>" in result


def test_how_unknown_raises():
    with pytest.raises(ValueError):
        Db("/Ethnicity/2024/LK/XLSX")
