import json
import logging
import sys

from lanka_data import Db


class ColorFormatter(logging.Formatter):
    GREY = "\033[90m"
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: GREY,
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
    }

    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelno)
        return f"{color}{msg}{self.RESET}" if color else msg


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter("[%(name)s] %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


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
