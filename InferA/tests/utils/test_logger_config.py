import os
import logging
import tempfile
import shutil
from unittest import mock

import pytest

import src.utils.logger_config as logger_config


@pytest.fixture(autouse=True)
def cleanup_log_handlers():
    # Clear handlers before each test
    yield
    root_logger = logging.getLogger()
    root_logger.handlers.clear()


@mock.patch("src.utils.logger_config.os.makedirs")
@mock.patch("src.utils.logger_config.logging.FileHandler")
def test_setup_logger_file_only(mock_file_handler, mock_makedirs, capsys):
    logger_config.setup_logger(enable_logging=True, enable_debug=False, enable_console_logging=False)

    captured = capsys.readouterr()
    assert "[LOGGING ENABLED] Logging to" in captured.out
    mock_makedirs.assert_called_once_with("logs", exist_ok=True)
    assert mock_file_handler.call_count == 1


@mock.patch("src.utils.logger_config.os.makedirs")
@mock.patch("src.utils.logger_config.logging.FileHandler")
@mock.patch("src.utils.logger_config.logging.StreamHandler")
def test_setup_logger_with_console(mock_stream_handler, mock_file_handler, mock_makedirs, capsys):
    logger_config.setup_logger(enable_logging=True, enable_debug=False, enable_console_logging=True)

    captured = capsys.readouterr()
    assert "Logging to" in captured.out
    assert "Also logging to console" in captured.out
    assert mock_file_handler.called
    assert mock_stream_handler.called


@mock.patch("src.utils.logger_config.os.makedirs")
@mock.patch("src.utils.logger_config.logging.StreamHandler")
def test_setup_logger_console_only(mock_stream_handler, mock_makedirs, capsys):
    logger_config.setup_logger(enable_logging=False, enable_debug=True, enable_console_logging=True)

    captured = capsys.readouterr()
    assert "Logging to console only" in captured.out
    assert mock_stream_handler.called


def test_setup_logger_null_handler():
    logger_config.setup_logger(enable_logging=False, enable_debug=False, enable_console_logging=False)
    root_logger = logging.getLogger()
    assert any(isinstance(h, logging.NullHandler) for h in root_logger.handlers)


def test_get_logger_returns_named_instance():
    logger = logger_config.get_logger("my_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "my_logger"
