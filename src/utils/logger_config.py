"""
Custom Logger Configuration Module

Provides:
- A custom TRACE logging level below DEBUG
- Custom console formatter to simplify INFO logs
- File logging with separate files for all logs and INFO-only logs
- Configurable options for console output verbosity and enabling TRACE level

How to use:
    import logger_config
    logger = logger_config.setup_logger(session="mysession", print_debug_to_console=True, enable_trace=False)
    logger.info("Logging initialized.")
"""

import logging
import os
from datetime import datetime

# Define custom TRACE level
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def trace(self, message, *args, **kwargs):
    """Add trace() method to all loggers for the TRACE level."""
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)

logging.Logger.trace = trace  # Monkey-patch Logger class



class ConsoleFormatter(logging.Formatter):
    """
    Custom formatter for console logs.

    INFO messages are printed plainly without the '[INFO]' prefix,
    while other levels include their level name in brackets.
    """
    def format(self, record):
        if record.levelno == logging.INFO:
            return record.getMessage()  # No [INFO] prefix
        else:
            return f"[{record.levelname}] {record.getMessage()}"

def setup_logger(session: str, print_debug_to_console: bool = False, enable_trace: bool = False):
    """
    Sets up logging with the following features:
    - Log files stored in ./logs/, created if missing.
    - One log file per run, timestamped.
    - Separate file for all logs and one file for INFO-only logs.
    - Console logging level and trace enablement configurable.

    Args:
        session (str): Identifier for this logging session; used to name the INFO-only log file.
        print_debug_to_console (bool): If True, prints DEBUG (or TRACE if enabled) logs to console.
                                       Otherwise, console prints only INFO and above.
        enable_trace (bool): If True, enables the custom TRACE level for all handlers.

    Returns:
        logging.Logger: The root logger configured with handlers.
    
    **User Configuration Points:**
        - Change `session` to label your INFO log file (e.g. "experiment1").
        - Toggle `print_debug_to_console` to control console verbosity.
        - Toggle `enable_trace` to enable very verbose TRACE logging across files and console.
        - Modify `LOG_DIR` if you want logs saved somewhere else.
        - Customize formats inside the function if different timestamp or message formats are desired.
    """
    LOG_DIR = "logs"    # <-- Change this to customize logs directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Timestamped log file for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = timestamp + ".log"
    log_path = os.path.join(LOG_DIR, log_filename)

    # Separate INFO-level log file named by session
    info_filename = session + "_INFO.log"
    info_log_path = os.path.join(LOG_DIR, info_filename)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()    # Remove any existing handlers

    # Set root logger level based on trace flag
    root_logger.setLevel(TRACE_LEVEL_NUM if enable_trace else logging.DEBUG)

    # File handler logs all messages including TRACE (if enabled)
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

    # Console handler with custom formatting and configurable level
    console_handler = logging.StreamHandler()
    if print_debug_to_console:
        console_level = TRACE_LEVEL_NUM if enable_trace else logging.DEBUG
    else:
        console_level = logging.INFO
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)

    return root_logger