"""
Tests for hispecdrp.utils.logging
"""
import logging
import pytest
from hispecdrp.utils.logging import (
    LogConfig,
    RunIdFilter,
    PackageCaptureFilter,
    ROOT_LOGGER,
    LOG_CAPTURE_PACKAGES,
    RUN_ID,
    _get_level,
    logging_context,
    setup_default_logging,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolate_loggers():
    """Snapshot and restore all managed loggers around each test."""
    names = [ROOT_LOGGER] + list(LOG_CAPTURE_PACKAGES)
    saved = {}
    for name in names:
        lg = logging.getLogger(name)
        saved[name] = (list(lg.handlers), lg.level, lg.propagate)
    yield
    for name, (handlers, level, propagate) in saved.items():
        lg = logging.getLogger(name)
        lg.handlers = handlers
        lg.setLevel(level)
        lg.propagate = propagate


# ── Helpers ───────────────────────────────────────────────────────────────────

class TestGetLevel:
    def test_string_upper(self):
        assert _get_level("INFO") == logging.INFO

    def test_string_lower(self):
        assert _get_level("debug") == logging.DEBUG

    def test_int_passthrough(self):
        assert _get_level(logging.WARNING) == logging.WARNING
        assert _get_level(30) == 30

    def test_unknown_defaults_to_info(self):
        assert _get_level("NOTAREAL") == logging.INFO


# ── Filters ───────────────────────────────────────────────────────────────────

class TestRunIdFilter:
    def test_attaches_run_id(self):
        f = RunIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert f.filter(record) is True
        assert record.run_id == RUN_ID


class TestPackageCaptureFilter:
    def test_accepts_hispecdrp(self):
        f = PackageCaptureFilter()
        record = logging.LogRecord(f"{ROOT_LOGGER}.sub", logging.INFO, "", 0, "msg", (), None)
        assert f.filter(record) is True

    def test_accepts_capture_packages(self):
        f = PackageCaptureFilter()
        for pkg in LOG_CAPTURE_PACKAGES:
            record = logging.LogRecord(f"{pkg}.sub", logging.INFO, "", 0, "msg", (), None)
            assert f.filter(record) is True

    def test_rejects_other(self):
        f = PackageCaptureFilter()
        record = logging.LogRecord("other.package", logging.INFO, "", 0, "msg", (), None)
        assert f.filter(record) is False


# ── LogConfig ─────────────────────────────────────────────────────────────────

class TestLogConfig:
    def test_level_from_string(self):
        cfg = LogConfig(level="WARNING")
        assert cfg.level == logging.WARNING

    def test_level_from_int(self):
        cfg = LogConfig(level=logging.DEBUG)
        assert cfg.level == logging.DEBUG

    def test_make_handlers_stdout_only(self):
        cfg = LogConfig(level="INFO", enable_stdout=True, enable_file=False)
        handlers = cfg.make_handlers()
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)

    def test_make_handlers_file_only(self, tmp_path):
        cfg = LogConfig(level="INFO", enable_stdout=False, enable_file=True, log_dir=str(tmp_path))
        handlers = cfg.make_handlers()
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.FileHandler)
        for h in handlers:
            h.close()

    def test_make_handlers_both(self, tmp_path):
        cfg = LogConfig(level="INFO", enable_stdout=True, enable_file=True, log_dir=str(tmp_path))
        handlers = cfg.make_handlers()
        assert len(handlers) == 2
        for h in handlers:
            h.close()

    def test_make_handlers_none(self):
        cfg = LogConfig(level="INFO", enable_stdout=False, enable_file=False)
        assert cfg.make_handlers() == []


# ── logging_context ───────────────────────────────────────────────────────────

class TestLoggingContext:
    def test_stdout(self, capsys):
        cfg = {"level": "INFO", "enable_stdout": True, "enable_file": False}
        log = logging.getLogger(f"{ROOT_LOGGER}.ctx_test")
        with logging_context(cfg):
            log.info("ctx stdout message")
        assert "ctx stdout message" in capsys.readouterr().out

    def test_file(self, tmp_path):
        cfg = {"level": "INFO", "enable_stdout": False, "enable_file": True, "log_dir": str(tmp_path)}
        log = logging.getLogger(f"{ROOT_LOGGER}.ctx_file_test")
        with logging_context(cfg):
            log.info("ctx file message")
        log_files = list(tmp_path.glob("*.txt"))
        assert len(log_files) == 1
        assert "ctx file message" in log_files[0].read_text()

    def test_level_filtering(self, tmp_path):
        cfg = {"level": "WARNING", "enable_stdout": False, "enable_file": True, "log_dir": str(tmp_path)}
        log = logging.getLogger(f"{ROOT_LOGGER}.level_test")
        with logging_context(cfg):
            log.debug("debug msg")
            log.info("info msg")
            log.warning("warning msg")
            log.error("error msg")
        content = list(tmp_path.glob("*.txt"))[0].read_text()
        assert "debug msg" not in content
        assert "info msg" not in content
        assert "warning msg" in content
        assert "error msg" in content

    def test_captures_koa_middleware(self, tmp_path):
        cfg = {"level": "INFO", "enable_stdout": False, "enable_file": True, "log_dir": str(tmp_path)}
        koa_log = logging.getLogger("koa_middleware.ctx_test")
        hispec_log = logging.getLogger(f"{ROOT_LOGGER}.ctx_test")
        with logging_context(cfg):
            hispec_log.info("hispec message")
            koa_log.info("koa message")
        content = list(tmp_path.glob("*.txt"))[0].read_text()
        assert "hispec message" in content
        assert "koa message" in content

    def test_does_not_capture_other_packages(self, tmp_path):
        cfg = {"level": "INFO", "enable_stdout": False, "enable_file": True, "log_dir": str(tmp_path)}
        other_log = logging.getLogger("other.package")
        hispec_log = logging.getLogger(f"{ROOT_LOGGER}.ctx_test")
        with logging_context(cfg):
            hispec_log.info("hispec message")
            other_log.info("other message")
        content = list(tmp_path.glob("*.txt"))[0].read_text()
        assert "hispec message" in content
        assert "other message" not in content

    def test_restores_logger_state(self):
        root_log = logging.getLogger(ROOT_LOGGER)
        root_log.handlers = []
        setup_default_logging()
        handlers_before = list(root_log.handlers)
        level_before = root_log.level
        propagate_before = root_log.propagate
        cfg = {"level": "INFO", "enable_stdout": True, "enable_file": False}
        with logging_context(cfg):
            pass
        assert root_log.handlers == handlers_before
        assert root_log.level == level_before
        assert root_log.propagate == propagate_before

    def test_nested_is_noop_outer_wins(self, capsys):
        outer_cfg = {"level": "INFO", "enable_stdout": True, "enable_file": False}
        inner_cfg = {"level": "WARNING", "enable_stdout": True, "enable_file": False}
        log = logging.getLogger(f"{ROOT_LOGGER}.reentrant_test")
        with logging_context(outer_cfg):
            with logging_context(inner_cfg):
                log.info("info inside nested")  # outer=INFO so this should appear
        assert "info inside nested" in capsys.readouterr().out

    def test_applied_sentinel_cleared_after_exit(self):
        cfg = {"level": "INFO", "enable_stdout": False, "enable_file": False}
        with logging_context(cfg):
            assert LogConfig.applied is not None
        assert LogConfig.applied is None

    def test_applied_sentinel_cleared_after_exception(self):
        cfg = {"level": "INFO", "enable_stdout": False, "enable_file": False}
        with pytest.raises(RuntimeError):
            with logging_context(cfg):
                raise RuntimeError("boom")
        assert LogConfig.applied is None


# ── setup_default_logging ─────────────────────────────────────────────────────

class TestSetupDefaultLogging:
    def test_installs_handler_on_hispecdrp(self):
        logging.getLogger(ROOT_LOGGER).handlers = []
        setup_default_logging()
        assert len(logging.getLogger(ROOT_LOGGER).handlers) > 0

    def test_installs_handler_on_capture_packages(self):
        for name in [ROOT_LOGGER] + list(LOG_CAPTURE_PACKAGES):
            logging.getLogger(name).handlers = []
        setup_default_logging()
        for pkg in LOG_CAPTURE_PACKAGES:
            assert len(logging.getLogger(pkg).handlers) > 0

    def test_idempotent(self):
        logging.getLogger(ROOT_LOGGER).handlers = []
        setup_default_logging()
        count = len(logging.getLogger(ROOT_LOGGER).handlers)
        setup_default_logging()
        assert len(logging.getLogger(ROOT_LOGGER).handlers) == count

    def test_stdout_output(self, capsys):
        log = logging.getLogger(ROOT_LOGGER)
        log.handlers = []
        setup_default_logging()
        log.info("outside context message")
        assert "outside context message" in capsys.readouterr().out

    def test_capture_packages_output(self, capsys):
        for name in [ROOT_LOGGER] + list(LOG_CAPTURE_PACKAGES):
            logging.getLogger(name).handlers = []
        setup_default_logging()
        logging.getLogger("koa_middleware.default_test").info("koa outside context")
        assert "koa outside context" in capsys.readouterr().out