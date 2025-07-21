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

def setup_logger(print_debug_to_console: bool = False, enable_trace: bool = False):
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    log_path = os.path.join(LOG_DIR, log_filename)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File logs everything including TRACE
    root_logger.setLevel(TRACE_LEVEL_NUM if enable_trace else logging.DEBUG)

    # File handler
    file_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(TRACE_LEVEL_NUM if enable_trace else logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

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