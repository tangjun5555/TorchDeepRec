# -*- coding: utf-8 -*-

from enum import Enum
from dataclasses import dataclass

import torch


class Mode(Enum):
    """
    Train/Evaluate/Predict Mode.
    """
    TRAIN = 1
    EVALUATE = 2
    PREDICT = 3


@dataclass
class ParsedData:
    name: str
    values: torch.Tensor
