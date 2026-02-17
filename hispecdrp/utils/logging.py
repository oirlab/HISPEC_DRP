import logging
import inspect
import multiprocessing as mp
from logging.handlers import (
    QueueHandler,
    QueueListener,
    RotatingFileHandler,
)
from contextlib import contextmanager
from datetime import datetime
import os
import sys

_QUEUE = None
_LISTENER = None
_CONFIGURED = False

RUN_ID = datetime.now().strftime("%Y%m%dT%H%M%S")

ROOT_LOGGER = "hispecdrp"

LOG_CAPTURE_PACKAGES = [
    "koa_middleware",
]

DEFAULT_FILENAME_PREFIX = "hispecdrp_log"

def _get_level(level):
    if isinstance(level, int):
        return level
    return logging._nameToLevel.get(str(level).upper(), logging.INFO)

def _make_formatter():
    return logging.Formatter(
        "[%(asctime)s]"
        "[%(name)s]"
        "[%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

class RunIdFilter(logging.Filter):
    def filter(self, record):
        record.run_id = RUN_ID
        return True

class PackageCaptureFilter(logging.Filter):
    def filter(self, record):
        for pkg in LOG_CAPTURE_PACKAGES:
            if record.name.startswith(pkg):
                return True
        return record.name.startswith(ROOT_LOGGER)

class LogConfig:
    def __init__(
        self,
        *,
        level=None,
        enable_stdout=None,
        enable_file=None,
        log_dir=None,
        filename_prefix=DEFAULT_FILENAME_PREFIX,
        rotate=False,
        max_bytes=50_000_000,
        backup_count=5,
    ):
        if level is None:
            level = os.getenv("HISPECDRP_LOG_LEVEL", "INFO")
        self.level = _get_level(level)

        if enable_stdout is None:
            enable_stdout = os.getenv("HISPECDRP_LOG_ENABLE_STDOUT", "true").lower() in {"1", "true", "yes", "on"}
        self.enable_stdout = enable_stdout

        if enable_file is None:
            enable_file = os.getenv("HISPECDRP_LOG_ENABLE_FILE", "false").lower() in {"1", "true", "yes", "on"}
        self.enable_file = enable_file

        if log_dir is None and self.enable_file:
            log_dir = os.getenv("HISPECDRP_LOG_DIR")
            if log_dir is None:
                log_dir = os.path.join(
                    os.path.expanduser("~"),
                    "hispecdrp_logs"
                ) + os.sep

        self.filename_prefix = filename_prefix

        self.log_dir = log_dir
        self.rotate = rotate
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        self.handlers = []
        self._previous_levels = {}

        self.formatter = _make_formatter()
        self.runid_filter = RunIdFilter()
        self.capture_filter = PackageCaptureFilter()

    def _make_file_handler(self, path):
        if self.rotate:
            return RotatingFileHandler(
                path,
                mode="a",
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
            )
        return logging.FileHandler(path, mode="a")

    def _setup_handlers(self):
        handlers = []

        if self.enable_stdout:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(self.formatter)
            sh.addFilter(self.runid_filter)
            sh.addFilter(self.capture_filter)
            handlers.append(sh)

        if self.enable_file and self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            path = os.path.join(
                self.log_dir,
                f"{self.filename_prefix}_{RUN_ID}_{os.getpid()}.txt"
            )
            fh = self._make_file_handler(path)
            fh.setFormatter(self.formatter)
            fh.addFilter(self.runid_filter)
            fh.addFilter(self.capture_filter)
            handlers.append(fh)

        self.handlers = handlers

    def apply(self, log_names=None):
        if not self.handlers:
            self._setup_handlers()

        if log_names is None:
            log_names = [ROOT_LOGGER]

        for name in log_names:
            logger = logging.getLogger(name)
            self._previous_levels[name] = logger.level
            logger.setLevel(self.level)

            for h in self.handlers:
                if h not in logger.handlers:
                    logger.addHandler(h)

        # Attach handlers to already-existing captured packages
        for lname in logging.root.manager.loggerDict:
            if any(lname.startswith(p) for p in LOG_CAPTURE_PACKAGES):
                logger = logging.getLogger(lname)
                for h in self.handlers:
                    if h not in logger.handlers:
                        logger.addHandler(h)

    def undo(self, log_names=None, close_handlers=True):
        if log_names is None:
            log_names = [ROOT_LOGGER]

        for name in log_names:
            logger = logging.getLogger(name)
            for h in self.handlers:
                logger.removeHandler(h)

            if name in self._previous_levels:
                logger.setLevel(self._previous_levels[name])

        # Remove handlers from captured package loggers
        for lname in logging.root.manager.loggerDict:
            if any(lname.startswith(p) for p in LOG_CAPTURE_PACKAGES):
                logger = logging.getLogger(lname)
                for h in self.handlers:
                    logger.removeHandler(h)

        # Flush and close handlers after removing from all loggers
        for h in self.handlers:
            h.flush()
            if close_handlers:
                h.close()

        self.handlers = []

    @contextmanager
    def context(self, log_names=None):
        self.apply(log_names)
        try:
            yield
        finally:
            self.undo(log_names)

# Queue for Multiprocessing
def _start_listener(queue, config: LogConfig):
    global _LISTENER

    if not config.handlers:
        config._setup_handlers()

    _LISTENER = QueueListener(queue, *config.handlers)
    _LISTENER.start()

def _stop_listener():
    global _LISTENER
    if _LISTENER is not None:
        _LISTENER.stop()
        _LISTENER = None

def _configure_process(queue, level):
    root = logging.getLogger()
    root.setLevel(_get_level(level))
    root.handlers.clear()
    root.addHandler(QueueHandler(queue))

# Main API

@contextmanager
def logging_context(config_dict, queue=None):
    global _QUEUE, _CONFIGURED

    if not _CONFIGURED:
        if queue is None:
            _QUEUE = mp.Queue(-1)
        else:
            _QUEUE = queue

        log_cfg = config_dict.get("logging", {})

        cfg = LogConfig(
            level=log_cfg.get("level", "INFO"),
            enable_stdout=log_cfg.get("enable_stdout"),
            enable_file=log_cfg.get("enable_file"),
            log_dir=log_cfg.get("log_dir"),
            filename_prefix=log_cfg.get(
                "filename_prefix", DEFAULT_FILENAME_PREFIX
            ),
            rotate=log_cfg.get("rotate", False),
            max_bytes=log_cfg.get("max_bytes", 50_000_000),
            backup_count=log_cfg.get("backup_count", 5),
        )

        _configure_process(_QUEUE, cfg.level)

        if queue is None:
            _start_listener(_QUEUE, cfg)

        _CONFIGURED = True

    try:
        yield
    finally:
        _stop_listener()
        _CONFIGURED = False

def get_queue():
    return _QUEUE