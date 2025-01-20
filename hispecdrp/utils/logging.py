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


def get_logger(primitive : "BasePrimitive") -> logging.Logger:
    """
    Args:
        primitive (BasePrimitive): Name of logger, which will be shown in each log msg.
        log_path (str): (optional) path to the logfile if different than whats defined in the configuration file
    """

    #if not primitive.LOGGER['enable']:

    # Initialize the logger
    logger_name = primitive.__class__.__name__
    logger = logging.getLogger(logger_name)
    logger.propagate = False

    # Get logger config
    log_cfg = primitive.config.get('LOGGER')

    # Check if writing to a file
    if log_cfg is not None:
        log_path = log_cfg.get('log_path')
        if log_path is None:
            log_path = primitive.LOGGER.get('log_path')
    else:
        log_path = None

    # Log level (what gets logged and what doesn't)
    if log_cfg is not None:
        log_lvl = log_cfg.get('level', 'warning')
    else:
        log_lvl = 'warning'
    logger.setLevel(get_level(log_lvl))

    # Format
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
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