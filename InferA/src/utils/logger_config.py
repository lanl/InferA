# logger_config.py
import logging
import os
from datetime import datetime

# Config flags
def setup_logger(enable_logging: bool, enable_debug: bool, enable_console_logging: bool = False):
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    log_path = os.path.join(LOG_DIR, log_filename)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.DEBUG if enable_debug else logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")

    if enable_logging:
        # Always add file handler if file logging enabled
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        if enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        print(f"[LOGGING ENABLED] Logging to {log_filename} (file)")
        if enable_console_logging:
            print("[LOGGING ENABLED] Also logging to console")
    else:
        # No file logging, but if console logging is True, add stream handler only
        if enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            print("[LOGGING ENABLED] Logging to console only (no file)")
        else:
            # No logging anywhere
            root_logger.addHandler(logging.NullHandler())

    return root_logger