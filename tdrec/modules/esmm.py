# -*- coding: utf-8 -*-

from typing import Dict, Any, List

import torch
from tdrec.modules.mlp import MLP
from tdrec.modules.mmoe import MMoE


class ESMM(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 num_task: int,
                 task_mlp: Dict[str, Any] = None,
                 mmoe: Dict[str, Any] = None,
                 ):
        super().__init__()

        self.num_task = num_task

        self.task_mlps = None
        if task_mlp is not None:
            self.task_mlps = torch.nn.ModuleList()
            for i in range(self.num_task):
                self.task_mlps.append(MLP(in_features=in_features, **task_mlp))

        self.mmoe = None
        if mmoe is not None:
            self.mmoe = MMoE(in_features=in_features, **mmoe)
