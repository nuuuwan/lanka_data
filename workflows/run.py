import json
import sys

from lanka_data import Db


def main(cmd):
    db = Db(cmd)
    result = db.run(open_images=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
