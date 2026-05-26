import json
import sys

from lanka_data import Db
from workflows.custom_logging import setup_logging


def main(cmd):
    setup_logging()
    db = Db(cmd)
    result = db.run_unsafe(do_open_images=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]))
