import os

from lanka_data import Db
from utils_future import JSONFile

DATA_DIR = os.path.join("tests", "data")
CMDS_FILE = os.path.join(DATA_DIR, "cmds.json")


def main():
    cmds = JSONFile(CMDS_FILE).read()

    # Remove stale numbered files before regenerating
    for f in os.listdir(DATA_DIR):
        if f.endswith(".json") and f != "cmds.json":
            os.remove(os.path.join(DATA_DIR, f))

    for i, cmd in enumerate(cmds):
        actual_output = Db(cmd).run(do_open_images=False, do_use_cache=False)
        path = os.path.join(DATA_DIR, f"{i:03d}.json")
        JSONFile(path).write({"cmd": cmd, "expected_output": actual_output})


if __name__ == "__main__":
    main()
