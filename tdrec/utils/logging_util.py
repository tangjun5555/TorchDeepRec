# -*- coding: utf-8 -*-

import logging
import time

logger = logging.getLogger("tdrec")
logger.setLevel(logging.INFO)


class ProgressLogger:
    """
    Logger with iterate speed.
    """

    def __init__(
        self, desc: str, start_n: int = -1, mininterval: float = 1, miniters: int = 0
    ) -> None:
        self._desc = desc
        self._last_time = time.time()
        self._last_n = start_n
        self._mininterval = mininterval
        self._miniters = miniters

    def set_description(self, desc: str) -> None:
        """Set logger description."""
        self._desc = desc

    def log(self, n: int, suffix: str = "") -> None:
        """Log iteration."""
        dn = n - self._last_n
        if dn > self._miniters:
            cur_time = time.time()
            dt = cur_time - self._last_time
            if dt > self._mininterval:
                logger.info(f"{self._desc}: {n}it [{dn / dt:.2f}it/s] {suffix}")
                self._last_time = cur_time
                self._last_n = n
