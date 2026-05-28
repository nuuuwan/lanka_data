import os

from lanka_data import Db
from utils_future import JSONFile

DATA_DIR = os.path.join("tests", "data")


def get_all_cmds():
    return [
        f[:-5].replace(".", "/")
        for f in sorted(os.listdir(DATA_DIR))
        if f.endswith(".json")
    ]


def main():
    for cmd in get_all_cmds():
        db = Db(cmd)
        actual_output = db.run(do_open_images=False, do_use_cache=False)
        filename = cmd.replace("/", ".") + ".json"
        JSONFile(os.path.join(DATA_DIR, filename)).write(actual_output)


if __name__ == "__main__":
    main()
