import logging
import multiprocessing as mp
import multiprocessing.queues
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
_SUPPRESSED_PKG_HANDLERS: dict[str, dict] = {}

RUN_ID = datetime.now().strftime("%Y%m%dT%H%M%S")
ROOT_LOGGER = "hispecdrp"
LOG_CAPTURE_PACKAGES = [
    "koa_middleware",
]
DEFAULT_FILENAME_PREFIX = "hispecdrp_log"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_level(level):
    if isinstance(level, int):
        return level
    return logging._nameToLevel.get(str(level).upper(), logging.INFO)


def _make_formatter():
    return logging.Formatter(
        "[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _suppress_package_handlers() -> dict[str, dict]:
    """
    Remove all handlers from each captured-package logger and ensure
    propagate=True so records flow to the root QueueHandler only.

    koa_middleware may have no handlers right now, but this guards against
    duplication if it ever adds its own in a future version or environment.

    Returns the saved state so it can be restored later.
    """
    suppressed: dict[str, dict] = {}
    for pkg in LOG_CAPTURE_PACKAGES:
        logger = logging.getLogger(pkg)
        suppressed[pkg] = {
            "handlers": logger.handlers[:],
            "propagate": logger.propagate,
            "level": logger.level,
        }
        logger.handlers = []
        logger.propagate = True
        # Ensure the logger doesn't silently filter records before propagation.
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.DEBUG)
    return suppressed


def _restore_package_handlers(suppressed: dict[str, dict]) -> None:
    """Restore previously suppressed package-logger handlers."""
    for pkg, state in suppressed.items():
        logger = logging.getLogger(pkg)
        logger.handlers = state["handlers"]
        logger.propagate = state["propagate"]
        logger.setLevel(state["level"])


# ── Filters ───────────────────────────────────────────────────────────────────

class RunIdFilter(logging.Filter):
    def filter(self, record):
        record.run_id = RUN_ID
        return True


class PackageCaptureFilter(logging.Filter):
    """Only pass through records from hispecdrp or captured packages."""
    def filter(self, record):
        if record.name.startswith(ROOT_LOGGER):
            return True
        return any(record.name.startswith(pkg) for pkg in LOG_CAPTURE_PACKAGES)


# ── LogConfig ─────────────────────────────────────────────────────────────────

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
            enable_stdout = os.getenv("HISPECDRP_LOG_ENABLE_STDOUT", "true").lower() in {
                "1", "true", "yes", "on"
            }
        self.enable_stdout = enable_stdout

        if enable_file is None:
            enable_file = os.getenv("HISPECDRP_LOG_ENABLE_FILE", "false").lower() in {
                "1", "true", "yes", "on"
            }
        self.enable_file = enable_file

        if log_dir is None and self.enable_file:
            log_dir = os.getenv("HISPECDRP_LOG_DIR", os.path.join(os.getcwd(), "logs"))

        self.filename_prefix = filename_prefix
        self.log_dir = log_dir
        self.rotate = rotate
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.handlers: list[logging.Handler] = []
        self._previous_levels: dict[str, int] = {}
        self.formatter = _make_formatter()
        self.runid_filter = RunIdFilter()
        self.capture_filter = PackageCaptureFilter()

    def _make_file_handler(self, path: str) -> logging.Handler:
        if self.rotate:
            return RotatingFileHandler(
                path,
                mode="a",
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
            )
        return logging.FileHandler(path, mode="a")

    def _setup_handlers(self) -> None:
        handlers: list[logging.Handler] = []

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
                f"{self.filename_prefix}_{RUN_ID}_{os.getpid()}.txt",
            )
            fh = self._make_file_handler(path)
            fh.setFormatter(self.formatter)
            fh.addFilter(self.runid_filter)
            fh.addFilter(self.capture_filter)
            handlers.append(fh)

        self.handlers = handlers

    def apply(self, log_names: list[str] | None = None) -> None:
        if not self.handlers:
            self._setup_handlers()

        if log_names is None:
            log_names = [ROOT_LOGGER]

        # Suppress BEFORE adding our handlers, not after
        global _SUPPRESSED_PKG_HANDLERS
        _SUPPRESSED_PKG_HANDLERS = _suppress_package_handlers()

        for name in log_names:
            logger = logging.getLogger(name)
            self._previous_levels[name] = logger.level
            logger.setLevel(self.level)
            for h in self.handlers:
                if h not in logger.handlers:
                    logger.addHandler(h)

        # Directly attach our handlers to captured packages
        for pkg in LOG_CAPTURE_PACKAGES:
            pkg_logger = logging.getLogger(pkg)
            pkg_logger.setLevel(self.level)
            for h in self.handlers:
                if h not in pkg_logger.handlers:
                    pkg_logger.addHandler(h)

    def undo(self, log_names: list[str] | None = None, close_handlers: bool = True) -> None:
        if log_names is None:
            log_names = [ROOT_LOGGER]

        for name in log_names:
            logger = logging.getLogger(name)
            for h in self.handlers:
                logger.removeHandler(h)
            if name in self._previous_levels:
                logger.setLevel(self._previous_levels[name])

        # Clean up package loggers too
        for pkg in LOG_CAPTURE_PACKAGES:
            pkg_logger = logging.getLogger(pkg)
            for h in self.handlers:
                pkg_logger.removeHandler(h)

        global _SUPPRESSED_PKG_HANDLERS
        _restore_package_handlers(_SUPPRESSED_PKG_HANDLERS)
        _SUPPRESSED_PKG_HANDLERS = {}

        for h in self.handlers:
            h.flush()
            if close_handlers:
                h.close()
        self.handlers = []

    @contextmanager
    def context(self, log_names: list[str] | None = None):
        self.apply(log_names)
        try:
            yield
        finally:
            self.undo(log_names)


# ── Queue / Multiprocessing ───────────────────────────────────────────────────

def _start_listener(queue: mp.queues.Queue, config: LogConfig) -> None:
    global _LISTENER
    if not config.handlers:
        config._setup_handlers()
    _LISTENER = QueueListener(queue, *config.handlers, respect_handler_level=True)
    _LISTENER.start()


def _stop_listener() -> None:
    global _LISTENER
    if _LISTENER is not None:
        _LISTENER.stop()
        _LISTENER = None


def _configure_process(queue: mp.queues.Queue, level) -> None:
    """
    Point the root logger exclusively at the queue so every record from
    every package (including koa_middleware) travels through one path.
    Suppresses any handlers the captured packages may have installed on
    their own loggers to prevent double-emission.
    """
    root = logging.getLogger()
    root.setLevel(_get_level(level))
    root.handlers.clear()
    root.addHandler(QueueHandler(queue))


def _restore_root_logger(handlers, level):
    """Restore root logger handlers and level (used when exiting logging_context)."""
    root = logging.getLogger()
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)


# Main API
    global _SUPPRESSED_PKG_HANDLERS
    _SUPPRESSED_PKG_HANDLERS = _suppress_package_handlers()


# ── Main public API ───────────────────────────────────────────────────────────

@contextmanager
def logging_context(config_dict: dict, queue: mp.queues.Queue | None = None):
    global _QUEUE, _CONFIGURED

    # Save root logger state so we can restore it when exiting (so script/caller
    # loggers continue to work after the pipeline runs).
    root = logging.getLogger()
    _saved_root_handlers = list(root.handlers)
    _saved_root_level = root.level

    # Save and clear the hispecdrp logger's default handlers so they don't
    # fire alongside the queue-based handlers during pipeline execution.
    hispecdrp_logger = logging.getLogger(ROOT_LOGGER)
    _saved_hispecdrp_handlers = list(hispecdrp_logger.handlers)
    _saved_hispecdrp_level = hispecdrp_logger.level
    hispecdrp_logger.handlers = []

    if not _CONFIGURED:
        if queue is None:
            _QUEUE = mp.Queue(-1)
        else:
            _QUEUE = queue

        cfg = LogConfig(
            level=config_dict.get("level", "INFO"),
            enable_stdout=config_dict.get("enable_stdout"),
            enable_file=config_dict.get("enable_file"),
            log_dir=config_dict.get("log_dir"),
            filename_prefix=config_dict.get("filename_prefix", DEFAULT_FILENAME_PREFIX),
            rotate=config_dict.get("rotate", False),
            max_bytes=config_dict.get("max_bytes", 50_000_000),
            backup_count=config_dict.get("backup_count", 5),
        )

        _configure_process(_QUEUE, cfg.level)

        if queue is None:
            _start_listener(_QUEUE, cfg)

        _CONFIGURED = True

    try:
        yield
    finally:
        _restore_package_handlers(_SUPPRESSED_PKG_HANDLERS)
        _stop_listener()
        _CONFIGURED = False
        _restore_root_logger(_saved_root_handlers, _saved_root_level)
        hispecdrp_logger.handlers = _saved_hispecdrp_handlers
        hispecdrp_logger.setLevel(_saved_hispecdrp_level)

def setup_default_logging() -> None:
    """
    Install a default stdout handler on the ``hispecdrp`` logger and each
    package in ``LOG_CAPTURE_PACKAGES`` so logs emitted outside of any
    ``logging_context`` are visible.  Called once at package import time.
    Skipped if the ``hispecdrp`` logger already has handlers (avoids
    duplicates when called multiple times or when the user has pre-configured
    logging before importing hispecdrp).
    """
    hispecdrp_logger = logging.getLogger(ROOT_LOGGER)
    if hispecdrp_logger.handlers:
        return

    from .config import get_config_path, read_config_file
    try:
        cfg_dict = read_config_file(get_config_path("logging.yaml"))
    except FileNotFoundError:
        cfg_dict = {}

    cfg = LogConfig(
        level=cfg_dict.get("level", "INFO"),
        enable_stdout=cfg_dict.get("enable_stdout", True),
        enable_file=False,  # never write files as part of the import-time default
    )
    cfg._setup_handlers()

    hispecdrp_logger.setLevel(cfg.level)
    for h in cfg.handlers:
        hispecdrp_logger.addHandler(h)

    # Also attach to each captured package so their records are visible
    # outside of logging_context (mirrors what logging_context does at runtime).
    for pkg in LOG_CAPTURE_PACKAGES:
        pkg_logger = logging.getLogger(pkg)
        pkg_logger.setLevel(cfg.level)
        for h in cfg.handlers:
            if h not in pkg_logger.handlers:
                pkg_logger.addHandler(h)

def get_queue():
    return _QUEUE