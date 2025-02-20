# -*- coding: utf-8 -*-

import torch


def test_grouped_auc():
    from tdrec.metrics.grouped_auc import GroupedAUC
    metric = GroupedAUC()
    preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    target = torch.tensor([1, 0, 1, 0, 0, 0, 1, 1])
    group_id = torch.tensor([1, 2, 2, 1, 3, 3, 4, 4])
    metric.update(preds, target, group_id)
    value = metric.compute()
    torch.testing.assert_close(value, torch.tensor(0.5))

