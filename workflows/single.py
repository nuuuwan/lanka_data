import json
import sys

from lanka_data import Command


def main(cmd):
    db = Command.from_str(cmd)
    result = db.run_unsafe(do_open_images=True, do_use_cache=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
