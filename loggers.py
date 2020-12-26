import logging

from rich.logging import RichHandler


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
        logger.addHandler(RichHandler(rich_tracebacks=True))

    return logger
