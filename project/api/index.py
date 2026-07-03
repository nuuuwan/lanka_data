import json
from http.server import BaseHTTPRequestHandler

from lanka_data import CommandRunner


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.replace("/api/", "").strip("/")
        try:
            result = CommandRunner.run(path)
            body = json.dumps(result)
            status = 200
        except Exception as e:
            body = json.dumps({"error": str(e)})
            status = 400

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())
