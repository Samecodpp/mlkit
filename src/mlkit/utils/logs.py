import logging
import sys

HANDLER_NAME = "mlkit-console"


def setup_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger("mlkit")
    logger.setLevel(level)
    logger.propagate = False

    if any(handler.get_name() == HANDLER_NAME for handler in logger.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
