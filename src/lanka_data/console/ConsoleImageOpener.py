import subprocess
import sys


class ConsoleImageOpener:
    @staticmethod
    def open(output):
        result = output.get("result")
        if not result or "image_path" not in result:
            return
        image_path = result["image_path"]
        command = ConsoleImageOpener.command_for_platform(image_path)
        if command:
            ConsoleImageOpener.run(command)

    @staticmethod
    def run(command):
        try:
            subprocess.run(command, check=False)
        except OSError:
            return

    @staticmethod
    def command_for_platform(image_path):
        if sys.platform == "darwin":
            return ["open", image_path]
        if sys.platform.startswith("linux"):
            return ["xdg-open", image_path]
        return None
