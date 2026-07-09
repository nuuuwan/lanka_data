import os
from http.server import BaseHTTPRequestHandler

from lanka_data.api.command.Command import Command
from lanka_data.api.command_errors.CommandError import CommandError
from lanka_data.datasets.command.CommandRunner import CommandRunner
from lanka_data.visual.plot.Plot import Plot
from api.HandlerResponseMixin import HandlerResponseMixin

Plot.DIR_OUTPUT = os.environ.get("LANKA_DATA_OUTPUT_DIR", Plot.DIR_OUTPUT)

OUTPUTS = (("/Image.png", "image_path", "image/png"),)
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
        for suffix, key, content_type in OUTPUTS:
            if path.endswith(suffix.strip("/")):
                command_str = path[: -len(suffix)]
                self._serve_output(command_str, key, content_type)
                return
        self._serve_json(path)

    def _get_result(self, command_str):
        self._validate_command(command_str)
        return CommandRunner.run(command_str)

    def _get_safe_output_path(self, result, key):
        output_path = (result.get("result") or {}).get(key)
        self._validate_safe_image_path(output_path)
        if not output_path or not os.path.exists(output_path):
            raise CommandError("Image not found")
        return output_path

    def _read_output(self, command_str, key):
        result = self._get_result(command_str)
        output_path = self._get_safe_output_path(result, key)
        with open(output_path, "rb") as f:
            return f.read()

    def _serve_output(self, command_str, key, content_type):
        data = self._run_safely(lambda: self._read_output(command_str, key))
        if data is not None:
            self._write_image(data, CACHE_CONTROL_IMAGE, content_type)

    def _hide_image_path(self, path, result):
        inner = result.get("result")
        if not self._has_image_path(inner):
            return result
        self._validate_output_paths(inner)
        return {**result, "result": self._to_public_image_result(path, inner)}

    def _validate_output_paths(self, inner):
        for _, key, _ in OUTPUTS:
            self._validate_safe_image_path(inner.get(key))

    def _has_image_path(self, inner):
        return isinstance(inner, dict) and "image_path" in inner

    def _to_public_image_result(self, path, inner):
        output_keys = {key for _, key, _ in OUTPUTS}
        new_inner = {k: v for k, v in inner.items() if k not in output_keys}
        for suffix, key, _ in OUTPUTS:
            if key in inner:
                url_key = key.replace("_path", "_url")
                new_inner[url_key] = self._public_url(f"{path}{suffix}")
        return new_inner

    def _build_json_result(self, path):
        return self._hide_image_path(path, self._get_result(path))

    def _serve_json(self, path):
        result = self._run_safely(lambda: self._build_json_result(path))
        if result is not None:
            self._write_json(
                200,
                result,
                CACHE_CONTROL_JSON,
                self._correction_headers(result),
            )

    @staticmethod
    def _correction_headers(result):
        if not (result or {}).get("is_corrected"):
            return {}
        reason = result.get("correction_reason", "")
        return {
            "X-Lanka-Corrected": "true",
            "Warning": f'299 - "{reason}"',
        }

    def _public_url(self, path):
        host = self.headers.get("Host", "")
        scheme = self.headers.get("X-Forwarded-Proto", "https")
        return f"{scheme}://{host}/{path}"
