import json
import os
from http.server import BaseHTTPRequestHandler

from lanka_data import CommandRunner

IMAGE_SUFFIX = "/Image.png"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0].replace("/api/", "").strip("/")

        if path.endswith(IMAGE_SUFFIX.strip("/")):
            self._serve_image(path[: -len(IMAGE_SUFFIX)])
            return

        self._serve_json(path)

    def _serve_image(self, command_str):
        try:
            result = CommandRunner.run(command_str)
            image_path = (result.get("result") or {}).get("image_path")
            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError("Image not found")
            with open(image_path, "rb") as f:
                data = f.read()
        except Exception as e:
            self._write_json(400, {"error": str(e)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_json(self, path):
        try:
            result = CommandRunner.run(path)
            inner = result.get("result")
            if isinstance(inner, dict) and "image_path" in inner:
                del inner["image_path"]
                inner["image_url"] = self._public_url(f"{path}{IMAGE_SUFFIX}")
            self._write_json(200, result)
        except Exception as e:
            self._write_json(400, {"error": str(e)})

    def _public_url(self, path):
        host = self.headers.get("Host", "")
        scheme = self.headers.get("X-Forwarded-Proto", "https")
        return f"{scheme}://{host}/{path}"

    def _write_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
