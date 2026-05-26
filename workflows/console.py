import json

from lanka_data import Db
from workflows.custom_logging import setup_logging


def main():
    setup_logging()

    print("")
    print("/Where/What/When/How")
    print("")
    while True:
        cmd = input("> /")

        if cmd in ["x", "q"]:
            break

        output = Db(cmd).run(open_images=True)
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
