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

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return Batch(
            features={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.features.items()
            },
            labels={
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.labels.items()
            },
            sample_weight=self.sample_weight.to(device=device, non_blocking=non_blocking),
        )
