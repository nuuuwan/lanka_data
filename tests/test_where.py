"""Tests for Where — region code matching and level resolution."""

import pytest

from lanka_data.core import Where

# --- Exact match ---


def test_exact_match():
    assert Where("LK").matches("LK")


def test_exact_no_match():
    assert not Where("LK").matches("LK-1")
    assert not Where("LK-11").matches("LK-1")


# --- Administrative levels ---


def test_provinces():
    w = Where("LK:Provinces")
    assert w.matches("LK-1")
    assert w.matches("LK-9")
    assert not w.matches("LK-11")
    assert not w.matches("LK")


def test_districts_national():
    w = Where("LK:Districts")
    assert w.matches("LK-11")
    assert w.matches("LK-52")
    assert not w.matches("LK-1")
    assert not w.matches("LK")


def test_districts_under_province():
    w = Where("LK-1:Districts")
    assert w.matches("LK-11")
    assert w.matches("LK-12")
    assert not w.matches("LK-21")


# --- Electoral levels ---


def test_eds_alias():
    w = Where("LK:EDs")
    assert w.level == "ElectoralDistricts"
    assert w.matches("EC-01")
    assert w.matches("EC-22")
    assert not w.matches("EC-01A")


def test_pds_alias():
    w = Where("EC-01:PDs")
    assert w.level == "PollingDivisions"
    assert w.matches("EC-01A")
    assert w.matches("EC-01Z")
    assert not w.matches("EC-02A")


# --- DSDs ---


def test_dsds_national():
    w = Where("LK:DSDs")
    assert w.matches("LK-1101")
    assert w.matches("LK-9101")
    assert not w.matches("LK-11")
    assert not w.matches("LK1101")  # old no-dash format must not match


def test_dsds_under_district():
    w = Where("LK-11:DSDs")
    assert w.matches("LK-1101")
    assert w.matches("LK-1109")
    assert not w.matches("LK-1201")


def test_dsds_under_province():
    w = Where("LK-1:DSDs")
    assert w.matches("LK-1101")
    assert w.matches("LK-1201")
    assert not w.matches("LK-2101")


# --- GNDs ---


def test_gnds_national():
    w = Where("LK:GNDs")
    assert w.matches("LK-1101001")
    assert w.matches("LK-9101099")
    assert not w.matches("LK-1101")  # DSD code, not GND


# --- Unknown level raises ---


def test_unknown_level_raises():
    with pytest.raises(ValueError):
        Where("LK:Municipalities")
