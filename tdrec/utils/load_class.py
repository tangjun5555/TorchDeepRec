# -*- coding: utf-8 -*-

import pydoc
import traceback


def load_by_path(path):
    path = path.strip()
    if path == "" or path is None:
        return None
    if "lambda" in path:
        return eval(path)
    try:
        return pydoc.locate(path)
    except pydoc.ErrorDuringImport:
        print("load %s failed: %s" % (path, traceback.format_exc()))
        return None
