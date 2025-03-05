# -*- coding: utf-8 -*-

from tdrec.utils.logging_util import logger


def to_int_list(value: str, sep: str="|", default_value: int = 0):
    split = value.split(sep)
    res = []
    for x in split:
        try:
            res.append(int(x))
        except Exception:
            logger.warn(f"{value} is not int list format.")
            res.append(default_value)
    return res
