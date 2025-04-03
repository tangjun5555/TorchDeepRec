# -*- coding: utf-8 -*-

from typing import Dict, Any, List

import torch
from tdrec.modules.mlp import MLP


class MMoE(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 expert_mlp: Dict[str, Any],
                 num_expert: int,
                 num_task: int,
                 gate_mlp: Dict[str, Any] = None,
                 ):
        super().__init__()
        self.num_expert = num_expert
        self.num_task = num_task

        self.expert_mlps = torch.nn.ModuleList(
            [MLP(in_features=in_features, **expert_mlp) for _ in range(num_expert)]
        )

        gate_final_in = in_features
        self.has_gate_mlp = False
        if gate_mlp is not None:
            self.has_gate_mlp = True
            self.gate_mlps = torch.nn.ModuleList(
                [MLP(in_features=in_features, **gate_mlp) for _ in range(num_task)]
            )
            gate_final_in = self.gate_mlps[0].hidden_units[-1]
        self.gate_finals = torch.nn.ModuleList(
            [torch.nn.Linear(gate_final_in, num_expert) for _ in range(num_task)]
        )

    @property
    def output_dim(self) -> int:
        """
        Output dimension of the module.
        """
        return self.expert_mlps[0].hidden_units[-1]

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        expert_fea_list = []
        for i in range(self.num_expert):
            expert_fea_list.append(self.expert_mlps[i](inputs))
        expert_feas = torch.stack(expert_fea_list, dim=1)

        result = []
        for i in range(self.num_task):
            if self.has_gate_mlp:
                gate = self.gate_mlps[i](inputs)
            else:
                gate = inputs
            gate = self.gate_finals[i](gate)
            gate = torch.softmax(gate, dim=1).unsqueeze(1)
            task_input = torch.matmul(gate, expert_feas).squeeze(1)
            result.append(task_input)
        return result
