import hashlib
import os

from lanka_data import Db
from utils_future import JSONFile

DATA_DIR = os.path.join("tests", "data")
CMDS_FILE = os.path.join("tests", "cmds.json")


def cmd_to_hash(cmd):
    return hashlib.md5(cmd.encode()).hexdigest()[:8]


def main():
    cmds = JSONFile(CMDS_FILE).read()

    # Remove stale data files before regenerating
    for f in os.listdir(DATA_DIR):
        if f.endswith(".json"):
            os.remove(os.path.join(DATA_DIR, f))

    for cmd in cmds:
        actual_output = Db(cmd).run(do_open_images=False, do_use_cache=False)
        path = os.path.join(DATA_DIR, f"{cmd_to_hash(cmd)}.json")
        JSONFile(path).write({"cmd": cmd, "expected_output": actual_output})


if __name__ == "__main__":
    main()
