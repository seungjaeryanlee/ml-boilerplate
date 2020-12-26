from loggers import get_default_logger


def main():
    logger = get_default_logger()
    logger.info("Hello World!")


if __name__ == "__main__":
    main()
