import json
import logging
import sys
import time

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

        t_start = time.time()
        if cmd in ["x", "q"]:
            break

        try:
            result = Db(cmd).run()
            query_time_ms = int((time.time() - t_start) * 1000)
            print(
                json.dumps(
                    dict(
                        result=result,
                        query_time_ms=query_time_ms,
                    ),
                    indent=2,
                )
            )
        except Exception as e:
            query_time_ms = int((time.time() - t_start) * 1000)
            print(
                json.dumps(
                    dict(
                        error=str(e),
                        query_time_ms=query_time_ms,
                    ),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
