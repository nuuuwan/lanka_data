import json
import sys

from lanka_data import Db


def main(cmd):
    db = Db(cmd)
    result = db.run_unsafe()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
