# -*- coding: utf-8 -*-

import pydoc
import traceback
from abc import ABCMeta


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


def register_class(class_map, class_name, cls):
    """
    Register a class into class_map.
    Args:
        class_map: class register map.
        class_name: name of the class.
        cls: a class.
    """
    assert class_name not in class_map or class_map[class_name] == cls, (
        f"confilict class {cls} , "
        f"{class_name} is already register to be {class_map[class_name]}"
    )
    print('register class %s' % class_name)
    class_map[class_name] = cls


def get_register_class_meta(class_map):
    """
    Get a meta class with registry.
    Args:
        class_map: class register map.
    Return:
        a meta class with registry.
    """

    class RegisterABCMeta(ABCMeta):
        def __new__(mcs, name, bases, attrs):
            newclass = super(RegisterABCMeta, mcs).__new__(mcs, name, bases, attrs)
            register_class(class_map, name, newclass)
            @classmethod
            def create_class(cls, name):
                if name in class_map:
                    return class_map[name]
                else:
                    raise Exception(
                        "Class %s is not registered. Available ones are %s"
                        % (name, list(class_map.keys()))
                    )
            newclass.create_class = create_class
            return newclass
    return RegisterABCMeta
