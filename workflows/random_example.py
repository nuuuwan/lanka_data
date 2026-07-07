import json
import os

from lanka_data import CommandConstructor, CommandRunner

if __name__ == "__main__":
    command = CommandConstructor.construct()
    output = CommandRunner.run(command)
    result = output["result"]
    if "image_path" in result:
        image_path = result["image_path"]
        os.system(f"open {image_path}")
    print(json.dumps(output, indent=4, ensure_ascii=False))
