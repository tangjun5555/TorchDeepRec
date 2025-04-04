# -*- coding: utf-8 -*-

from typing import Dict, Any, List

import torch
from tdrec.modules.mlp import MLP
from tdrec.modules.mmoe import MMoE


class ESMM(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 num_task: int,
                 mmoe: Dict[str, Any] = None,
                 task_mlp: Dict[str, Any] = None,
                 ):
        super().__init__()
        self.num_task = num_task
        self.output_dim = in_features

        self.mmoe = None
        if mmoe is not None:
            self.mmoe = MMoE(in_features=in_features, num_task=num_task, **mmoe)
            in_features = self.mmoe.output_dim
            self.output_dim = self.mmoe.output_dim

        self.task_mlps = None
        if task_mlp is not None:
            self.task_mlps = torch.nn.ModuleList()
            for i in range(self.num_task):
                self.task_mlps.append(MLP(in_features=in_features, **task_mlp))
            self.output_dim = self.task_mlps[0].output_dim

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        if self.mmoe is not None:
            task_input_list = self.mmoe(inputs)
        else:
            task_input_list = [inputs] * self.task_num

        result = []
        if self.task_mlps is not None:
            for i in range(self.num_task):
                result.append(self.task_mlps[i](task_input_list[i]))
        else:
            return task_input_list
        return result
