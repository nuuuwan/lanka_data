import subprocess
import sys


class ConsoleImageOpener:
    @staticmethod
    def open(output):
        result = output.get("result")
        if not result or "image_path" not in result:
            return
        image_path = result["image_path"]
        if sys.platform == "darwin":
            subprocess.run(["open", image_path], check=False)
