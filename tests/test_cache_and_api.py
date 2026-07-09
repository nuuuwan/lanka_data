import os

from lanka_data.command.CommandCache import CommandCache
from lanka_data.visual.plot.Plot import Plot
from project.api.index import handler


class TestCacheAndApi:
    def test_command_cache_evicts_oldest_entry(self):
        cache = CommandCache(max_size=2)
        cache.set("a", ({}, []))
        cache.set("b", ({}, []))
        cache.set("c", ({}, []))
        assert cache.get("a") is None
        assert cache.get("b") == ({}, [])
        assert cache.get("c") == ({}, [])

    def test_command_cache_drops_missing_image_entry(self, tmp_path):
        cache = CommandCache(max_size=2)
        image_path = tmp_path / "missing.png"
        cache.set("a", ({"image_path": str(image_path)}, []))
        assert cache.get("a") is None

    def test_api_image_path_must_stay_inside_output_dir(self):
        api = object.__new__(handler)
        safe_path = os.path.join(Plot.DIR_OUTPUT, "x", "Image.png")
        assert api._is_safe_image_path(safe_path)
        assert not api._is_safe_image_path("/tmp/other/Image.png")

    def test_api_public_result_exposes_image_and_svg_urls(self):
        api = object.__new__(handler)
        api.headers = {"Host": "example.com", "X-Forwarded-Proto": "https"}
        inner = {
            "image_path": os.path.join(Plot.DIR_OUTPUT, "c", "Image.png"),
            "other": 1,
        }
        result = api._hide_image_path(
            "Religion/2024/LK/Map", {"result": inner}
        )
        new_inner = result["result"]
        assert "image_path" not in new_inner
        assert "svg_path" not in new_inner
        assert new_inner["other"] == 1
        assert new_inner["image_url"].endswith(
            "/Religion/2024/LK/Map/Image.png"
        )
        assert "svg_url" not in new_inner
