# -*- coding: utf-8 -*-

from typing import Any
import torch
from torchmetrics import Metric
from torchmetrics.functional.classification.auroc import _binary_auroc_compute
from torchmetrics.utilities.data import dim_zero_cat


class GroupedAUC(Metric):
    """
    Grouped AUC.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("grouping_key", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor, grouping_key: torch.Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)
        self.grouping_key.append(grouping_key)

    def compute(self) -> torch.Tensor:
        grouping_key = dim_zero_cat(self.grouping_key)
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        sorted_grouping_key, indices = torch.sort(grouping_key)
        sorted_preds = preds[indices]
        sorted_target = target[indices]

        _, counts = torch.unique_consecutive(sorted_grouping_key, return_counts=True)
        counts = counts.tolist()

        grouped_preds = torch.split(sorted_preds, counts)
        grouped_target = torch.split(sorted_target, counts)

        aucs = []
        for preds, target in zip(grouped_preds, grouped_target):
            mean_target = torch.mean(target.to(torch.float32)).item()
            if mean_target > 0 and mean_target < 1:
                aucs.append(_binary_auroc_compute((preds, target), None))

        return torch.mean(torch.Tensor(aucs))
