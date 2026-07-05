import json
import traceback

from api.command.CommandError import CommandError

IMAGE_CONTENT_TYPE = "image/png"
JSON_CONTENT_TYPE = "application/json"


class HandlerResponseMixin:
    def _write_image(self, data, cache_control):
        self.send_response(200)
        self.send_header("Content-Type", IMAGE_CONTENT_TYPE)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(data)

    def _run_safely(self, func):
        try:
            return func()
        except CommandError as e:
            self._write_json(400, {"error": e.to_dict()})
        except Exception:
            traceback.print_exc()
            self._write_json(500, {"error": "Internal server error"})
        return None

    def _write_json(self, status, obj, cache_control="no-store"):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", JSON_CONTENT_TYPE)
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)
