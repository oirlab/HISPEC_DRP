import logging
import multiprocessing as mp
import multiprocessing.queues
from logging.handlers import (
    QueueHandler,
    QueueListener,
    RotatingFileHandler,
)
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import os
import sys

from .config import get_config_dir, load_config_file

_QUEUE: mp.queues.Queue | None = None
_LISTENER: QueueListener | None = None

RUN_ID = datetime.now().strftime("%Y%m%dT%H%M%S")
ROOT_LOGGER = "hispecdrp"
LOG_CAPTURE_PACKAGES = [
    "koa_middleware",
]
DEFAULT_FILENAME_PREFIX = "hispecdrp_log"

# The only loggers we ever touch. Children of these propagate naturally
# upward and fire exactly once on the top-level logger's single handler.
_MANAGED_LOGGERS = [ROOT_LOGGER] + LOG_CAPTURE_PACKAGES


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


def _make_stdout_handler(level: int) -> logging.StreamHandler:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_make_formatter())
    h.setLevel(level)
    return h


def _make_file_handler(
    level: int,
    log_dir: str,
    filename_prefix: str,
    rotate: bool,
    max_bytes: int,
    backup_count: int,
) -> logging.Handler:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{filename_prefix}_{RUN_ID}_{os.getpid()}.txt")
    if rotate:
        h = RotatingFileHandler(path, mode="a", maxBytes=max_bytes, backupCount=backup_count)
    else:
        h = logging.FileHandler(path, mode="a")
    h.setFormatter(_make_formatter())
    h.setLevel(level)
    return h


def _save_managed_loggers() -> dict:
    state = {}
    for name in _MANAGED_LOGGERS:
        lg = logging.getLogger(name)
        state[name] = {
            "handlers": list(lg.handlers),
            "level": lg.level,
            "propagate": lg.propagate,
        }
    return state


def _restore_managed_loggers(state: dict) -> None:
    for name, s in state.items():
        lg = logging.getLogger(name)
        lg.handlers = list(s["handlers"])
        lg.setLevel(s["level"])
        lg.propagate = s["propagate"]


def _install_handler_on_managed_loggers(handler: logging.Handler, level: int) -> None:
    """
    Replace all handlers on every managed logger with the given handler.
    Sets propagate=False so records stop here and never reach root.
    Child loggers (e.g. hispecdrp.processing_component) are untouched —
    they have no handlers and propagate=True by default, so their records
    bubble up to the managed logger and fire exactly once.
    """
    for name in _MANAGED_LOGGERS:
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.setLevel(level)
        lg.propagate = False


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
    # Class-level sentinel: set to the active LogConfig while a
    # logging_context is live, None otherwise. Inner logging_context
    # calls check this and become no-ops.
    applied: "LogConfig | None" = None

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

    def make_handlers(self) -> list[logging.Handler]:
        """Build the real output handlers (stdout and/or file)."""
        handlers = []
        if self.enable_stdout:
            handlers.append(_make_stdout_handler(self.level))
        if self.enable_file and self.log_dir:
            handlers.append(_make_file_handler(
                self.level, self.log_dir, self.filename_prefix,
                self.rotate, self.max_bytes, self.backup_count,
            ))
        return handlers


# ── Queue / Listener ─────────────────────────────────────────────────────────

def _start_listener(queue: mp.queues.Queue, handlers: list[logging.Handler]) -> None:
    global _LISTENER
    _LISTENER = QueueListener(queue, *handlers, respect_handler_level=True)
    _LISTENER.start()


def _stop_listener() -> None:
    global _LISTENER
    if _LISTENER is not None:
        _LISTENER.stop()  # drains queue and joins listener thread
        _LISTENER = None


def _close_queue() -> None:
    global _QUEUE
    if _QUEUE is not None:
        _QUEUE.close()
        _QUEUE.join_thread()  # stops feeder thread, prevents process hang
        _QUEUE = None


# ── Main public API ───────────────────────────────────────────────────────────

@contextmanager
def logging_context(config_dict: dict, queue: mp.queues.Queue | None = None):
    """
    The single entry point for all logging in hispecdrp.

    Outermost call owns the queue and listener for its full lifetime.
    Nested calls (e.g. a pipeline invoking a primitive that also calls
    logging_context) are no-ops — they yield immediately without touching
    the queue, so the outer context stays alive.
    """
    if LogConfig.applied is not None:
        yield
        return

    global _QUEUE

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

    _saved_state = _save_managed_loggers()

    _QUEUE = queue if queue is not None else mp.Queue(-1)
    real_handlers = cfg.make_handlers()
    _start_listener(_QUEUE, real_handlers)

    qh = QueueHandler(_QUEUE)
    qh.setLevel(cfg.level)
    _install_handler_on_managed_loggers(qh, cfg.level)

    LogConfig.applied = cfg

    try:
        yield
    finally:
        LogConfig.applied = None
        _stop_listener()   # drain first
        _close_queue()     # then kill feeder thread
        for h in real_handlers:
            h.flush()
            h.close()
        _restore_managed_loggers(_saved_state)


def setup_default_logging() -> None:
    """
    Install a default stdout handler on all managed loggers so records
    are visible outside any logging_context (e.g. at import time or in
    scripts). Idempotent — skipped if hispecdrp already has handlers.
    """
    hispecdrp_logger = logging.getLogger(ROOT_LOGGER)
    if hispecdrp_logger.handlers:
        return

    logging_config_path = os.path.join(get_config_dir(), "logging.yaml")

    try:
        cfg_dict = load_config_file(logging_config_path)
    except FileNotFoundError:
        cfg_dict = {}

    level = _get_level(cfg_dict.get("level", "INFO"))
    handler = _make_stdout_handler(level)
    _install_handler_on_managed_loggers(handler, level)


def get_queue() -> mp.queues.Queue | None:
    return _QUEUE


# Logging config for quiet_pipeline_logging: errors only, nothing on stdout.
QUIET_PIPELINE_LOG_CONFIG = {"level": "ERROR", "enable_stdout": False}


@contextmanager
def quiet_pipeline_logging():
    """Silence pipeline logging and stray progress prints.

    Stage-0 JWST ramp fitting still uses ``print`` for per-row progress, which
    floods notebook cells even when the logger is quiet. Redirect stdout/stderr
    to ``os.devnull`` for the duration so recovery tables stay readable.
    """
    with logging_context(QUIET_PIPELINE_LOG_CONFIG):
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
