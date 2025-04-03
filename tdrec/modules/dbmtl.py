# -*- coding: utf-8 -*-

from typing import Dict, Any, List

import torch
from tdrec.modules.mlp import MLP
from tdrec.modules.mmoe import MMoE


class DBMTL(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 task_names: List[str],
                 relation_names: List[str],
                 task_mlp: Dict[str, Any],
                 mmoe: Dict[str, Any] = None,
                 ):
        super().__init__()

        assert len(task_names) == len(relation_names)
        self.task_num = len(task_names)
        self.task_names = task_names

        self.mmoe = None
        if mmoe is not None:
            self.mmoe = MMoE(in_features=in_features, **mmoe)

        self.task_mlps = torch.nn.ModuleDict()
        for task_index in range(self.task_num):
            task_name = self.task_names[task_index]
            self.task_mlps[task_name] = MLP(in_features=in_features, **task_mlp)

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        if self.mmoe is not None:
            task_input_list = self.mmoe(inputs)
        else:
            task_input_list = [inputs] * self.task_num

        task_net = {}
        for task_index in range(self.task_num):
            task_name = self.task_names[task_index]
            task_net[task_name] = self.task_mlps[task_name](task_input_list[task_index])

        result = []
        for task_index in range(self.task_num):
            task_name = self.task_names[task_index]
            relation_name = self.relation_names[task_index]
            if relation_name:
                result.append(torch.cat([task_net[task_name], task_net[relation_name]], dim=1))
            else:
                result.append(task_net[task_name])
        return result
