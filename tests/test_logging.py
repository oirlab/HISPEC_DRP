"""
Tests for logging utilities in hispecdrp.utils.logging
"""
import logging
import pytest

from hispecdrp.utils.logging import (
    LogConfig,
    RunIdFilter,
    PackageCaptureFilter,
    ROOT_LOGGER,
    RUN_ID,
    _get_level,
)


@pytest.fixture
def logger():
    """Create a test logger"""
    logger_name = f"{ROOT_LOGGER}.test"
    log = logging.getLogger(logger_name)
    yield log
    # Cleanup
    for handler in log.handlers[:]:
        handler.close()
        log.removeHandler(handler)


class TestHandlerCombinations:
    """Test all 4 handler combinations: stdout/file on/off"""
    
    def test_stdout_only(self, logger, capsys):
        """Test stdout handler only"""
        config = LogConfig(level="INFO", enable_stdout=True, enable_file=False)
        config.apply([logger.name])
        
        logger.info("stdout message")
        captured = capsys.readouterr()
        
        assert "stdout message" in captured.out
        config.undo([logger.name])
    
    def test_file_only(self, logger, tmp_path):
        """Test file handler only"""
        config = LogConfig(level="INFO", enable_stdout=False, enable_file=True, log_dir=str(tmp_path))
        config.apply([logger.name])
        
        logger.info("file message")
        config.undo([logger.name])
        
        log_files = list(tmp_path.glob("*.txt"))
        assert len(log_files) == 1
        assert "file message" in log_files[0].read_text()
    
    def test_both_handlers(self, logger, tmp_path, capsys):
        """Test both stdout and file handlers"""
        config = LogConfig(level="INFO", enable_stdout=True, enable_file=True, log_dir=str(tmp_path))
        config.apply([logger.name])
        
        logger.info("both handlers")
        captured = capsys.readouterr()
        config.undo([logger.name])
        
        assert "both handlers" in captured.out
        log_files = list(tmp_path.glob("*.txt"))
        assert "both handlers" in log_files[0].read_text()
    
    def test_no_handlers(self, logger):
        """Test with both handlers disabled"""
        config = LogConfig(level="INFO", enable_stdout=False, enable_file=False)
        config.apply([logger.name])
        
        logger.info("no handlers")  # Should not raise error
        config.undo([logger.name])


class TestLogLevels:
    """Test log levels"""
    
    def test_log_levels(self, logger, tmp_path):
        """Test DEBUG, INFO, WARNING, ERROR levels"""
        config = LogConfig(level="WARNING", enable_stdout=False, enable_file=True, log_dir=str(tmp_path))
        config.apply([logger.name])
        
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")
        config.undo([logger.name])
        
        content = list(tmp_path.glob("*.txt"))[0].read_text()
        assert "debug" not in content
        assert "info" not in content
        assert "warning" in content
        assert "error" in content
    
    def test_get_level_helper(self):
        """Test _get_level helper function"""
        assert _get_level("INFO") == logging.INFO
        assert _get_level("debug") == logging.DEBUG
        assert _get_level(logging.WARNING) == logging.WARNING
        assert _get_level(30) == 30


class TestFilters:
    """Test logging filters"""
    
    def test_runid_filter(self):
        """Test RunIdFilter adds run_id"""
        filter_obj = RunIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        
        assert filter_obj.filter(record) is True
        assert record.run_id == RUN_ID
    
    def test_package_capture_filter(self):
        """Test PackageCaptureFilter accepts/rejects packages"""
        filter_obj = PackageCaptureFilter()
        
        # Accept hispecdrp
        record = logging.LogRecord(f"{ROOT_LOGGER}.test", logging.INFO, "", 0, "msg", (), None)
        assert filter_obj.filter(record) is True
        
        # Reject others
        record = logging.LogRecord("other.package", logging.INFO, "", 0, "msg", (), None)
        assert filter_obj.filter(record) is False


class TestPackageCapture:
    """Test captured packages like koa_middleware"""
    
    def test_koa_middleware_captured(self, tmp_path):
        """Test koa_middleware logs are captured"""
        koa_logger = logging.getLogger("koa_middleware.test")
        hispec_logger = logging.getLogger(f"{ROOT_LOGGER}.test")
        
        config = LogConfig(level="INFO", enable_stdout=False, enable_file=True, log_dir=str(tmp_path))
        config.apply([hispec_logger.name, "koa_middleware"])
        
        hispec_logger.info("hispec message")
        koa_logger.info("koa message")
        
        config.undo([hispec_logger.name])
        
        content = list(tmp_path.glob("*.txt"))[0].read_text()
        assert "hispec message" in content
        assert "koa message" in content
    
    def test_other_packages_filtered(self, tmp_path):
        """Test non-captured packages are filtered"""
        hispec_logger = logging.getLogger(f"{ROOT_LOGGER}.test")
        other_logger = logging.getLogger("other.package")
        
        config = LogConfig(level="INFO", enable_stdout=False, enable_file=True, log_dir=str(tmp_path))
        config.apply([hispec_logger.name])
        
        hispec_logger.info("hispec message")
        other_logger.info("other message")
        config.undo([hispec_logger.name])
        
        content = list(tmp_path.glob("*.txt"))[0].read_text()
        assert "hispec message" in content
        assert "other message" not in content
    
    def test_multiple_log_levels_captured(self, tmp_path):
        """Test multiple log levels for hispecdrp and basic capture for koa_middleware"""
        koa_logger = logging.getLogger("koa_middleware.test")
        hispec_logger = logging.getLogger(f"{ROOT_LOGGER}.test")
        
        config = LogConfig(level="INFO", enable_stdout=False, enable_file=True, log_dir=str(tmp_path))
        config.apply([hispec_logger.name, "koa_middleware"])
        
        # Test different log levels for hispecdrp
        hispec_logger.debug("hispec debug")
        hispec_logger.info("hispec info")
        hispec_logger.warning("hispec warning")
        hispec_logger.error("hispec error")
        
        # Test basic capture for koa_middleware
        koa_logger.info("koa info")
        
        config.undo([hispec_logger.name])
        
        content = list(tmp_path.glob("*.txt"))[0].read_text()
        
        # Debug should not appear (level is INFO)
        assert "hispec debug" not in content
        
        # INFO, WARNING, ERROR should all appear for hispecdrp
        assert "hispec info" in content
        assert "hispec warning" in content
        assert "hispec error" in content
        
        # koa_middleware should be captured
        assert "koa info" in content


class TestContextManager:
    """Test LogConfig context manager"""
    
    def test_context_cleanup(self, logger, tmp_path):
        """Test context manager cleans up handlers"""
        config = LogConfig(level="INFO", enable_stdout=False, enable_file=True, log_dir=str(tmp_path))
        initial_handlers = len(logger.handlers)
        
        with config.context([logger.name]):
            assert len(logger.handlers) > initial_handlers
            logger.info("in context")
        
        assert len(logger.handlers) == initial_handlers
        assert "in context" in list(tmp_path.glob("*.txt"))[0].read_text()
