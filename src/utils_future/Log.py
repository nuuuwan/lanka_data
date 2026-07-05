import logging


class CustomLoggingFormatter(logging.Formatter):
    GREY = "\033[90m"
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: GREY,
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }

    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelno)
        return f"{color}{msg}{self.RESET}" if color else msg


class Log(logging.Logger):
    def __init__(self, name: str = "unnamed", level: int = logging.DEBUG):
        super(Log, self).__init__(name, level)
        self.propagate = False

        formatter = CustomLoggingFormatter("[%(name)s] %(message)s")
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        self.handlers = [sh]  # noqa
