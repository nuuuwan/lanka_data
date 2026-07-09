import json
import os
import sys
import traceback

from lanka_data.api.command_errors.CommandError import CommandError

IMAGE_CONTENT_TYPE = "image/png"
JSON_CONTENT_TYPE = "application/json"
DEBUG = os.environ.get("LANKA_DATA_DEBUG", "") not in ("", "0", "false")


class HandlerResponseMixin:
    def _write_image(
        self, data, cache_control, content_type=IMAGE_CONTENT_TYPE
    ):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(data)

    def _log_error(self, e):
        path = getattr(self, "path", "?")
        print(
            f"[lanka_data] {type(e).__name__} on {path}: {e}",
            file=sys.stderr,
        )
        traceback.print_exc()

    def _error_payload(self, e):
        payload = {
            "type": type(e).__name__,
            "message": str(e),
            "path": getattr(self, "path", None),
        }
        if DEBUG:
            payload["traceback"] = traceback.format_exc().splitlines()
        return {"error": payload}

    def _run_safely(self, func):
        try:
            return func()
        except CommandError as e:
            self._write_json(400, {"error": e.to_dict()})
        except Exception as e:
            self._log_error(e)
            self._write_json(500, self._error_payload(e))
        return None

    def _write_json(
        self, status, obj, cache_control="no-store", extra_headers=None
    ):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", JSON_CONTENT_TYPE)
        self.send_header("Cache-Control", cache_control)
        for name, value in (extra_headers or {}).items():
            self.send_header(name, self._sanitize_header(value))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _sanitize_header(value):
        return str(value).replace("\r", "").replace("\n", "")
