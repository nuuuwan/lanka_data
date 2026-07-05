import os
from http.server import BaseHTTPRequestHandler

from lanka_data.api.command.Command import Command
from lanka_data.api.command.CommandError import CommandError
from lanka_data.datasets.command.CommandRunner import CommandRunner
from lanka_data.visual.plot.Plot import Plot
from project.api.HandlerResponseMixin import HandlerResponseMixin

IMAGE_SUFFIX = "/Image.png"
CACHE_CONTROL_JSON = (
    "public, max-age=300, s-maxage=86400, " "stale-while-revalidate=86400"
)
CACHE_CONTROL_IMAGE = "public, max-age=86400, s-maxage=31536000, immutable"


class handler(HandlerResponseMixin, BaseHTTPRequestHandler):
    def _validate_command(self, command_str):
        if command_str == "Help":
            return
        Command.from_str(command_str)

    def _is_safe_image_path(self, image_path):
        output_dir = os.path.realpath(Plot.DIR_OUTPUT)
        image_path = os.path.realpath(image_path)
        return os.path.commonpath([output_dir, image_path]) == output_dir

    def _validate_safe_image_path(self, image_path):
        if image_path and not self._is_safe_image_path(image_path):
            raise CommandError("Unsafe image path")

    def do_GET(self):
        path = self.path.split("?")[0].replace("/api/", "").strip("/")
        if path.endswith(IMAGE_SUFFIX.strip("/")):
            self._serve_image(path[: -len(IMAGE_SUFFIX)])
            return
        self._serve_json(path)

    def _get_result(self, command_str):
        self._validate_command(command_str)
        return CommandRunner.run(command_str)

    def _get_safe_image_path(self, result):
        image_path = (result.get("result") or {}).get("image_path")
        self._validate_safe_image_path(image_path)
        if not image_path or not os.path.exists(image_path):
            raise CommandError("Image not found")
        return image_path

    def _read_image(self, command_str):
        result = self._get_result(command_str)
        image_path = self._get_safe_image_path(result)
        with open(image_path, "rb") as f:
            return f.read()

    def _serve_image(self, command_str):
        data = self._run_safely(lambda: self._read_image(command_str))
        if data is not None:
            self._write_image(data, CACHE_CONTROL_IMAGE)

    def _hide_image_path(self, path, result):
        inner = result.get("result")
        image_path = (inner or {}).get("image_path")
        self._validate_safe_image_path(image_path)
        if not self._has_image_path(inner):
            return result
        return {**result, "result": self._to_public_image_result(path, inner)}

    def _has_image_path(self, inner):
        return isinstance(inner, dict) and "image_path" in inner

    def _to_public_image_result(self, path, inner):
        new_inner = {k: v for k, v in inner.items() if k != "image_path"}
        new_inner["image_url"] = self._public_url(f"{path}{IMAGE_SUFFIX}")
        return new_inner

    def _build_json_result(self, path):
        return self._hide_image_path(path, self._get_result(path))

    def _serve_json(self, path):
        result = self._run_safely(lambda: self._build_json_result(path))
        if result is not None:
            self._write_json(200, result, CACHE_CONTROL_JSON)

    def _public_url(self, path):
        host = self.headers.get("Host", "")
        scheme = self.headers.get("X-Forwarded-Proto", "https")
        return f"{scheme}://{host}/{path}"
