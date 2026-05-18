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


# --- LK district → EC mapping for election levels ---


def test_pds_under_lk_district():
    """LK-11 (Colombo admin district) should map to EC-01 polling divisions."""
    w = Where("LK-11:PDs")
    assert w.matches("EC-01A")
    assert w.matches("EC-01P")
    assert not w.matches("EC-02A")  # Gampaha PDs must not match
    assert not w.matches("EC-22A")  # Other district PDs must not match


def test_pds_under_lk_district_vanni():
    """LK-42 (Mannar) maps to Vanni electoral district EC-11."""
    w = Where("LK-42:PDs")
    assert w.matches("EC-11A")
    assert not w.matches("EC-10A")
    assert not w.matches("EC-12A")


def test_eds_under_lk_district():
    """LK-11 (Colombo admin district) should return only EC-01."""
    w = Where("LK-11:EDs")
    assert w.matches("EC-01")
    assert not w.matches("EC-02")
    assert not w.matches("EC-01A")  # PD, not ED


def test_eds_national_unchanged():
    """LK:EDs should still return all electoral districts."""
    w = Where("LK:EDs")
    assert w.matches("EC-01")
    assert w.matches("EC-22")


def test_pds_national_unchanged():
    """LK:PDs should still return all polling divisions."""
    w = Where("LK:PDs")
    assert w.matches("EC-01A")
    assert w.matches("EC-22Z")
