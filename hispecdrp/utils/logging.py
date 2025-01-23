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


def get_logger(name : str, config : Config) -> logging.Logger:
    """
    Args:
        name (str): Name of logger, which will be shown in each log msg.
        config (Config): Config object with logging settings.
    """
    if not config.get('logging', {}).get('enabled', True):
        return None

    # Initialize the logger
    logger = logging.getLogger(name)
    logger.propagate = False

    # Get logger config
    log_cfg = config.get('logging')

    # Check if writing to a file
    if not log_cfg:
        return logger
    
    # Log path
    log_path = log_cfg.get('log_path', None)

    # Log level (what gets logged and what doesn't)
    log_lvl = log_cfg.get('level', 'info')
    logger.setLevel(get_level(log_lvl))

    # Format
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    # Stream handler (NOTE: use verbose to set this?)
    s_handle = logging.StreamHandler()
    s_handle.setLevel(get_level(log_lvl))
    s_handle.setFormatter(formatter)
    logger.addHandler(s_handle)

    # Log file
    if log_path is not None:
        f_handle = logging.FileHandler(log_path, mode='w')
        f_handle.setLevel(get_level(log_lvl))
        f_handle.setFormatter(formatter)
        logger.addHandler(f_handle)

    return logger