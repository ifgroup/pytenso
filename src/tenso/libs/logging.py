# coding: utf-8
"""Interface to logging package.
"""
import logging
import os
import sys
from typing import Literal, Optional


class Logger(object):
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self,
                 filename: Optional[str] = None,
                 level: Literal['debug', 'info', 'warning', 'error',
                                'critical'] = 'info',
                 stream_fmt: Optional[str] = None,
                 file_fmt: str = '%(message)s'):
        if filename is None:
            # Use the same name of the main script as the default name
            filename = os.path.splitext(os.path.basename(
                sys.argv[0]))[0] + '.log'
        self._logger = logging.getLogger(filename)
        self._logger.setLevel(self.levels[level])
        if stream_fmt is not None:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter(stream_fmt))
            self._logger.addHandler(sh)
        th = logging.FileHandler(filename=filename, mode='w', encoding='utf-8')
        th.setFormatter(logging.Formatter(file_fmt))
        self._logger.addHandler(th)

    def info(self, message: str):
        """Log an info message."""
        self._logger.info(message)

    def debug(self, message: str):
        """Log a debug message."""
        self._logger.debug(message)

    def warning(self, message: str):
        """Log a warning message."""
        self._logger.warning(message)

    def error(self, message: str):
        """Log an error message."""
        self._logger.error(message)

    def critical(self, message: str):
        """Log a critical message."""
        self._logger.critical(message)

    def __del__(self):
        """Close the file handler when the logger is deleted."""
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self._logger.removeHandler(handler)
