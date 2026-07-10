import pytest

from lanka_data.api.command.Command import Command
from lanka_data.correction.Correction import Correction
from lanka_data.correction.CorrectionErrors import CorrectionLoopError, DestructiveCorrectionError
from lanka_data.correction.CorrectionPolicy import CorrectionPolicy
from lanka_data.correction.CorrectionPipeline import correct
from lanka_data.datasets.command.CommandRunner import CommandRunner

AUTO = CorrectionPolicy(destructive="auto")


def _corrections(command_str, policy=None):
    command = Command.from_str(command_str)
    corrected, corrections = correct(command, policy)
    return corrected.cmd_id, [c.to_dict() for c in corrections]


class _AlwaysRule:
    name = "always"
    field = "When"
    severity = "lossless"

    def applies(self, wc):
        return True

    def apply(self, wc):
        return wc, Correction(
            "When", self.name, "x", "y", "lossless", "loops forever"
        )


class TestCorrection:
    def test_empty_rejects_ranking_modifier(self):
        cmd_id, corrections = _corrections("Empty/2024/LK:district/Map:1st")
        assert cmd_id == "Empty/2024/LK:district/Map"
        assert corrections == [
            {
                "field": "How",
                "rule": "geometry_measurement_rejects_modifiers",
                "from": "Map:1st",
                "to": "Map",
                "severity": "lossless",
                "reason": (
                    "Empty binds no data; ranking modifiers require a"
                    " categorical measurement."
                ),
            }
        ]

    def test_scalar_rejects_diversity_modifier(self):
        cmd_id, corrections = _corrections(
            "Parliamentary/2024/LK/Map:DiversityPew"
        )
        assert cmd_id == "Parliamentary/2024/LK/Map"
        assert corrections == [
            {
                "field": "How",
                "rule": "modifier_requires_kind",
                "from": "Map:DiversityPew",
                "to": "Map",
                "severity": "lossless",
                "reason": (
                    "DiversityPew requires a categorical measurement;"
                    " Parliamentary is scalar."
                ),
            }
        ]

    def test_snap_observation_year(self):
        cmd_id, corrections = _corrections(
            "Religion/2013/LK:district/Map", AUTO
        )
        assert cmd_id == "Religion/2012/LK:district/Map"
        assert corrections == [
            {
                "field": "When",
                "rule": "snap_observation_year",
                "from": "2013",
                "to": "2012",
                "severity": "destructive",
                "reason": (
                    "2013 is not an observation year for Religion;"
                    " snapped to nearest (2012)."
                ),
            }
        ]

    def test_two_corrections_in_pipeline_order(self):
        cmd_id, corrections = _corrections("Empty/2013/LK/Map:1st", AUTO)
        assert cmd_id == "Empty/2012/LK/Map"
        assert [c["field"] for c in corrections] == ["When", "How"]
        assert [c["rule"] for c in corrections] == [
            "snap_observation_year",
            "geometry_measurement_rejects_modifiers",
        ]

    def test_destructive_interval_raises_under_default_policy(self):
        with pytest.raises(DestructiveCorrectionError):
            correct(
                Command.from_str("Religion/2012-2013/LK:district/Map:Change")
            )

    def test_destructive_interval_widens_under_auto(self):
        command = Command.from_str(
            "Religion/2012-2013/LK:district/Map:Change"
        )
        corrected, corrections = correct(command, AUTO)
        assert corrected.cmd_id == "Religion/2012-2024/LK:district/Map:Change"
        assert len(corrections) == 1
        assert corrections[0].rule == "snap_interval_endpoints"
        assert "widened" in corrections[0].reason

    def test_valid_command_emits_no_original(self):
        output = CommandRunner.run("Religion/2012/LK/JSON")
        assert output["is_corrected"] is False
        assert output["corrections"] == []
        assert "original_command_str" not in output
        assert "correction_reason" not in output

    def test_fixpoint_loop_guard(self):
        command = Command.from_str("Religion/2012/LK/Map")
        with pytest.raises(CorrectionLoopError):
            correct(command, rules=[_AlwaysRule()])


WHATS = ["Empty", "Religion", "Parliamentary", "RiverLen"]
WHENS = [
    "2012",
    "2013",
    "2024",
    "2012-2013",
    "2012-2024",
    "2001-2024",
    "2030",
]
WHERES = ["LK", "LK:district", "LK-42:district", "LK@20:district"]
HOWS = [
    "Map",
    "Map:1st",
    "Map:3rd",
    "Map:DiversityPew",
    "Map:Change",
    "BarChart",
    "JSON",
]


def _product():
    for what in WHATS:
        for when in WHENS:
            for where in WHERES:
                for how in HOWS:
                    yield f"{what}/{when}/{where}/{how}"


class TestCorrectionProperty:
    def _is_valid(self, command_str):
        try:
            Command.from_str(command_str)
            return True
        except ValueError:
            return False

    def test_range_is_exactly_valid_set(self):
        for command_str in _product():
            if not self._is_valid(command_str):
                continue
            self._check(command_str)

    def _check(self, command_str):
        command = Command.from_str(command_str)
        try:
            corrected, _ = correct(command, AUTO)
        except ValueError:
            return
        assert self._is_valid(corrected.cmd_id)
        again, corrections = correct(corrected, AUTO)
        assert again.cmd_id == corrected.cmd_id
        assert corrections == []
