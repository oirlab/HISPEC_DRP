import logging


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


def get_logger(primitive : "HISPECPrimitive") -> logging.Logger:
    """
    Args:
        primitive (Primitive): Name of logger, which will be shown in each log msg.
        log_path (str): (optional) path to the logfile if different than whats defined in the configuration file
    """
    logger_name = primitive.__class__.__name__
    logging.getLogger(logger_name)

    # Get logger config
    log_cfg = primitive.config['LOGGER']

    # Check if writing to a file
    if log_path is None:
        log_path = log_cfg.get('log_path')

    # Initialize the logger
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    # Log level (what gets logged and what doesn't)
    logger.setLevel(get_level(log_lvl))
    log_lvl = log_cfg.get('level', 'warning')

    # Format
    formatter = logging.Formatter("[%(name)s][%(levelname)s]:%(message)s")

    # Log file
    if log_path is not None:
        f_handle = logging.FileHandler(log_path, mode='w')
        f_handle.setLevel(get_level(log_lvl))
        f_handle.setFormatter(formatter)
        logger.addHandler(f_handle)
        s_handle = logging.StreamHandler()
        s_handle.setLevel(get_level(log_lvl))
        s_handle.setFormatter(formatter)
        logger.addHandler(s_handle)

    return logger