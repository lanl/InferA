# logger_config.py
import logging
import os
from datetime import datetime

# Define custom TRACE level
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)

logging.Logger.trace = trace  # Add trace() to all loggers

class ConsoleFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            return record.getMessage()  # No [INFO] prefix
        else:
            return f"[{record.levelname}] {record.getMessage()}"

def setup_logger(session: str, print_debug_to_console: bool = False, enable_trace: bool = False):
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = timestamp + ".log"
    log_path = os.path.join(LOG_DIR, log_filename)

    # Create separate file for INFO level logs
    info_filename = session + "_INFO.log"
    info_log_path = os.path.join(LOG_DIR, info_filename)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File logs everything including TRACE
    root_logger.setLevel(TRACE_LEVEL_NUM if enable_trace else logging.DEBUG)

    # File handler
    file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(TRACE_LEVEL_NUM if enable_trace else logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # INFO-only file handler - logs only INFO level and above
    info_file_formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    info_file_handler = logging.FileHandler(info_log_path, encoding="utf-8")
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(info_file_formatter)
    # Add a filter to only include INFO level logs
    info_file_handler.addFilter(lambda record: record.levelno == logging.INFO)
    root_logger.addHandler(info_file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    if print_debug_to_console:
        console_level = TRACE_LEVEL_NUM if enable_trace else logging.DEBUG
    else:
        console_level = logging.INFO
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)

    return root_logger