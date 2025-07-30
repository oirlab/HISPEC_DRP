import logging


def get_level(lvl : str) -> int:
    """
    Converts a string representation of a logging level to its integer equivalent.

    Args:
        lvl (str): The logging level as a string (e.g., 'debug', 'info', 'warning', 'error', 'critical').

    Returns:
        int: The corresponding integer logging level from the `logging` module.
    """
    lvl = lvl.lower()
    if lvl == 'debug': return logging.DEBUG
    elif lvl == 'info': return logging.INFO
    elif lvl == 'warning': return logging.WARNING
    elif lvl == 'error': return logging.ERROR
    elif lvl == 'critical': return logging.CRITICAL
    else: return logging.NOTSET

def get_logger(name: str, config : dict) -> logging.Logger:
    """
    Args:
        name (str): Name of logger, which will be shown in each log msg.
    """

    # Initialize the logger
    logger = logging.getLogger(name)
    log_cfg = config.get('logging', {})

    # Prevent propagation to the root logger
    logger.propagate = False

    # Determine log level
    log_lvl = log_cfg.get('level', 'info')
    logger.setLevel(get_level(log_lvl))

    # Desired log path (if any)
    log_path = log_cfg.get('log_path', None)

    # Define formatter
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s]: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Check existing handlers and update if necessary
    existing_handlers = {type(h): h for h in logger.handlers}
    
    # Stream handler
    if logging.StreamHandler not in existing_handlers:
        s_handle = logging.StreamHandler()
        s_handle.setFormatter(formatter)
        logger.addHandler(s_handle)
    else:
        s_handle = existing_handlers[logging.StreamHandler]
        s_handle.setLevel(get_level(log_lvl))
        s_handle.setFormatter(formatter)

    # File handler
    if log_path is not None:
        if logging.FileHandler not in existing_handlers or (
            isinstance(existing_handlers.get(logging.FileHandler), logging.FileHandler) and
            existing_handlers[logging.FileHandler].baseFilename != log_path
        ):
            # Remove the old file handler if it exists
            if logging.FileHandler in existing_handlers:
                logger.removeHandler(existing_handlers[logging.FileHandler])

            # Add a new file handler
            f_handle = logging.FileHandler(log_path, mode='w')
            f_handle.setFormatter(formatter)
            f_handle.setLevel(get_level(log_lvl))
            logger.addHandler(f_handle)
    elif logging.FileHandler in existing_handlers:
        # Remove file handler if log_path is no longer needed
        logger.removeHandler(existing_handlers[logging.FileHandler])

    return logger
