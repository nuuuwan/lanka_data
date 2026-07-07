import json
import os
import random

from lanka_data import CommandRunner, Example

if __name__ == "__main__":
    examples = Example.get_cmd_list()

    random.shuffle(examples)
    random_example = random.choice(examples)

    output = CommandRunner.run(random_example)
    result = output["result"]
    if "image_path" in result:
        image_path = result["image_path"]
        os.system(f"open {image_path}")
    print(json.dumps(output, indent=4, ensure_ascii=False))
