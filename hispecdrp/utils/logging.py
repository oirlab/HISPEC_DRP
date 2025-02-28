import logging
from .config import Config


def get_level(lvl : str) -> int:
    """
    Read the logging level (string) from config file and return 
    the corresponding logging level (int)
    """
    lvl = lvl.lower()
    if lvl == 'debug': return logging.DEBUG
    elif lvl == 'info': return logging.INFO
    elif lvl == 'warning': return logging.WARNING
    elif lvl == 'error': return logging.ERROR
    elif lvl == 'critical': return logging.CRITICAL
    else: return logging.NOTSET


def get_logger(name: str, config: Config) -> logging.Logger | None:
    """
    Args:
        name (str): Name of logger, which will be shown in each log msg.
        config (Config): Config object with logging settings.
    """
    if not config.get('logging', {}).get('enabled', True):
        return None

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
    if log_path:
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
