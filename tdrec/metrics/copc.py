# -*- coding: utf-8 -*-

from typing import Any

import torch
from torchmetrics import Metric


class COPC(Metric):
    """
    Click Over Predicted Click.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state("preds_sum", default=torch.zeros(1, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("target_sum", default=torch.zeros(1, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("weight_sum", default=torch.zeros(1, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, weight: torch.Tenso = None) -> None:
        self.preds_sum += torch.sum(preds)
        self.target_sum += torch.sum(target)
        if weight is None:
            self.weight_sum += 0.0
        else:
            self.weight_sum += torch.sum(weight)

    def compute(self) -> torch.Tensor:
        return self.preds_sum / (self.target_sum + self.weight_sum)
