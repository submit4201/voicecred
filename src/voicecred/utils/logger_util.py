import logging
from pathlib import Path


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Get a named logger with standard formatting.

    Example:
        logger = get_logger(__name__)
        logger.info("This is an info message.")

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    # ensure a dedicated log directory in a cross-platform-friendly way
    logs_dir = Path("log")
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we cannot create a log directory, fall back to streaming only
        logs_dir = None

    # avoid adding duplicate handlers if called repeatedly (common in tests)
    if logger.handlers:
        # still ensure level is set consistently
        logger.setLevel(level)
        return logger

    filehandler = None
    if logs_dir is not None:
        # create a file handler with explicit UTF-8 encoding
        filehandler = logging.FileHandler(str(logs_dir / f"{name}.log"), encoding="utf-8")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s     || %(name)s \n%(levelname)s   || %(message)s \n',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if filehandler is not None:
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)

    logger.setLevel(level)
    # stop passing records to the root logger (avoid duplicated messages)
    logger.propagate = False

    # log the initialization message after handlers are configured
    logger.info("\n"
        f"--------------------------------------------------\n"
        f"                    |'{name}' initialized with level {logging.getLevelName(level)}|\n"
        f"                    --------------------------------------------------\n"
    )
    return logger

logger = get_logger(__name__)