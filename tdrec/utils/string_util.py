# -*- coding: utf-8 -*-

def to_int_list(value: str, sep: str="|", default_value: int = 0):
    split = value.split(sep)
    res = []
    for x in split:
        try:
            res.append(int(x))
        except Exception:
            print(f"{value} is not int list format.")
            res.append(default_value)
    return res


def to_float_list(value: str, sep: str="|", default_value: float = 0.0):
    split = value.split(sep)
    res = []
    for x in split:
        try:
            res.append(float(x))
        except Exception:
            print(f"{value} is not float list format.")
            res.append(default_value)
    return res
