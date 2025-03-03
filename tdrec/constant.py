# -*- coding: utf-8 -*-

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict
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


@dataclass
class Batch:
    features: Dict[str, torch.Tensor] = field(default_factory=dict)
    labels: Dict[str, torch.Tensor] = field(default_factory=dict)
    sample_weight: torch.Tensor = field(default_factory=torch.Tensor)
