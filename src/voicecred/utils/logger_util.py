import logging

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Get a named logger with standard formatting.

    Example:
        logger = get_logger(__name__)
        logger.info("This is an info message.")

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        filehandler = logging.FileHandler(f"{name}.log")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

