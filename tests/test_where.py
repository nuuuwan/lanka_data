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


# --- Unknown level raises ---

def test_unknown_level_raises():
    with pytest.raises(ValueError):
        Where("LK:Municipalities")
