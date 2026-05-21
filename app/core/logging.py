import logging
import sys


def setup_logging(debug: bool = False, environment: str = "development") -> None:
    level = logging.DEBUG if debug else logging.INFO

    if environment == "production":
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logging.basicConfig(level=level, format=fmt, stream=sys.stdout, force=True)

    # Suppress noisy third-party output
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if debug else logging.WARNING
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
