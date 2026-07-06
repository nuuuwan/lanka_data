import os
import tempfile

import pytest
from PIL import Image

from lanka_data.command.Command import Command
from lanka_data.command.fields.How import How
from lanka_data.dataset.DatasetFactory import DatasetFactory
from lanka_data.visual.animation_visual.AnimationEncoder import \
    AnimationEncoder
from lanka_data.visual.animation_visual.AnimationVisual import AnimationVisual
from lanka_data.visual.VisualFactory import VisualFactory


class TestAnimationHow:
    def test_animation_bases_are_intervals(self):
        for base in How.ANIMATION_BASE_TO_FRAME_BASE:
            how = How(base)
            assert how.is_animation
            assert how.needs_interval

    def test_frame_how_carries_modifier(self):
        assert How("MapAnimation").frame_how == "Map"
        assert How("MapAnimation:1st").frame_how == "Map:1st"
        assert How("HexMapAnimation").frame_how == "HexMap"

    def test_non_animation_how_is_not_animation(self):
        assert not How("Map").is_animation
        assert How("Map").frame_how == "Map"

    def test_animation_requires_interval_when(self):
        Command.from_str("Religion/2001-2012-2024/LK:district/MapAnimation")
        with pytest.raises(Exception, match="interval"):
            Command.from_str("Religion/2024/LK:district/MapAnimation")


class TestAnimationFactory:
    def test_maps_animation_bases_to_animation_visual(self):
        for base in How.ANIMATION_BASE_TO_FRAME_BASE:
            assert VisualFactory._VISUAL_CLS[base] is AnimationVisual

    def test_builds_one_dataset_per_year(self, monkeypatch):
        calls = []

        class DummyDataset:
            def __init__(self, year):
                self.year = year

            def get_year(self):
                return self.year

        def fake_from_command(command):
            calls.append(command.when_cmd)
            return DummyDataset(command.when_cmd)

        monkeypatch.setattr(DatasetFactory, "from_command", fake_from_command)
        command = Command.from_str(
            "Religion/2001-2012-2024/LK:district/MapAnimation:1st"
        )
        datasets = DatasetFactory.list_from_command(command)
        assert calls == ["2001", "2012", "2024"]
        assert [d.panel_label for d in datasets] == ["2001", "2012", "2024"]


class TestAnimationEncoder:
    @staticmethod
    def _make_frames(output_dir, colors, size=(120, 80)):
        paths = []
        for i, color in enumerate(colors):
            path = os.path.join(output_dir, f"frame_{i}.png")
            Image.new("RGB", size, color).save(path)
            paths.append(path)
        return paths

    def test_encode_produces_multi_frame_gif(self):
        output_dir = tempfile.mkdtemp()
        paths = self._make_frames(
            output_dir, [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        )
        result = AnimationEncoder.encode(paths, output_dir)
        gif = Image.open(result["image_path"])
        assert result["image_path"].endswith(".gif")
        assert os.path.exists(result["image_path"])
        assert getattr(gif, "n_frames", 1) == 3

    def test_encode_normalizes_mismatched_frame_sizes(self):
        output_dir = tempfile.mkdtemp()
        paths = self._make_frames(output_dir, [(255, 0, 0)])
        odd_path = os.path.join(output_dir, "odd.png")
        Image.new("RGB", (60, 40), (0, 0, 0)).save(odd_path)
        paths.append(odd_path)
        result = AnimationEncoder.encode(paths, output_dir)
        assert getattr(Image.open(result["image_path"]), "n_frames", 1) == 2

    def test_encode_raises_without_frames(self):
        with pytest.raises(ValueError):
            AnimationEncoder.encode([], tempfile.mkdtemp())


class TestAnimationVisual:
    def test_build_sequences_frames_into_animation(self, monkeypatch):
        output_dir = tempfile.mkdtemp()

        class DummyDataset:
            def __init__(self, year):
                self.panel_label = year

            def get_year(self):
                return self.panel_label

        frame_paths = []
        for i, color in enumerate([(255, 0, 0), (0, 255, 0)]):
            path = os.path.join(output_dir, f"frame_{i}.png")
            Image.new("RGB", (100, 60), color).save(path)
            frame_paths.append(path)

        built = iter(frame_paths)
        monkeypatch.setattr(
            AnimationVisual,
            "_build_frame",
            lambda self, dataset: next(built),
        )

        command = Command.from_str(
            "Religion/2001-2012/LK:district/MapAnimation:1st"
        )
        visual = AnimationVisual(
            command=command,
            datasets=[DummyDataset("2001"), DummyDataset("2012")],
            how_cmd=command.how_cmd,
        )
        result = visual.build()
        assert result["frame_paths"] == frame_paths
        assert getattr(Image.open(result["image_path"]), "n_frames", 1) == 2
