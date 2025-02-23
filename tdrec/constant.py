# -*- coding: utf-8 -*-

from enum import Enum


class Mode(Enum):
    """
    Train/Evaluate/Predict Mode.
    """
    TRAIN = 1
    EVALUATE = 2
    PREDICT = 3
