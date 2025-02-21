# -*- coding: utf-8 -*-

import torch
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Batch:
    features: Dict[str, torch.Tensor] = field(default_factory=dict)
    labels: Dict[str, torch.Tensor] = field(default_factory=dict)
    sample_weight: torch.Tensor = field(default_factory=torch.Tensor)
