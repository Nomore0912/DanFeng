#!/usr/bin/env python
# -*- coding:utf-8 -*-
# transformers.utils.logging.py

import functools
import logging
import sys
import os
import threading
from typing import Optional


_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None
log_levels = {
    "detail": logging.DEBUG,  # will also print filename and line number
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
_default_log_level = logging.WARNING


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler.flush = sys.stderr.flush
        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        # if logging level is debug, we add pathname and lineno to formatter for easy debugging
        if os.getenv("TRANSFORMERS_VERBOSITY", None) == "detail":
            formatter = logging.Formatter("[%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s")
            _default_handler.setFormatter(formatter)

        library_root_logger.propagate = False


def _get_default_logging_level():
    """
    If TRANSFORMERS_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to `_default_log_level`
    """
    env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option TRANSFORMERS_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _get_library_name() -> str:
    return __name__.split(".")[0]


def get_logger(name=None):
    if name is None:
        name = _get_library_name()
    _configure_library_root_logger()
    return logging.getLogger(name)


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once
