# logging_config.py
"""
One place to configure logging. Every module calls get_logger(__name__)
instead of using print(), so output is consistent and can be redirected
(e.g. to a file or a log aggregator) without touching pipeline code.
"""
import logging
import sys, os
from app import config

_CONFIGURED = False
# log_file = config.LOG_FILE

def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED

    if not _CONFIGURED:
        # 1. Deferred Import to break any circular dependency chains
        from app import config

        log_file = config.LOG_FILE

        # 2. Defensively create logs folder structure if it is missing
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

        # 3. Configure handlers dynamically
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_file:
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s.%(funcName)s | %(message)s",
            datefmt="%H:%M:%S",
            handlers=handlers,
        )
        _CONFIGURED = True

    return logging.getLogger(name)