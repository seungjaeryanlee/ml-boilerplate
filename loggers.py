import logging


class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter with colors

    Part of this class is from https://stackoverflow.com/a/56944256/2577392.
    """

    format_ = "%(asctime)s - %(levelname)-8s - %(filename)s L%(lineno)d - %(message)s"

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey + format_ + reset,
        logging.INFO: grey + format_ + reset,
        logging.WARNING: yellow + format_ + reset,
        logging.ERROR: red + format_ + reset,
        logging.CRITICAL: bold_red + format_ + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        return formatter.format(record)


def get_default_logger(name="default", level=logging.DEBUG, to_file=False, to_console=True, filename="default.log"):
    """
    Return default logger to be used throughout the project
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if to_file:
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    if to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

    return logger
