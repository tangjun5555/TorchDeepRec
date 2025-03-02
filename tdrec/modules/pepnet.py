# -*- coding: utf-8 -*-

from typing import List

import torch
from tdrec.modules.mlp import Perceptron


class GateNU(torch.nn.Module):
    def __init__(self,
                 dense_in_features: int,
                 gate_in_features: int,
                 ):
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(dense_in_features + gate_in_features, dense_in_features)
        )
        self.gate.append(torch.nn.ReLU())
        self.gate.append(torch.nn.Linear(dense_in_features, dense_in_features))
        self.gate.append(torch.nn.Sigmoid())

    def forward(self, dense_inputs: torch.Tensor, gate_inputs: torch.Tensor) -> torch.Tensor:
        dense_inputs = dense_inputs.detach()
        inputs = torch.cat([dense_inputs, gate_inputs], dim=-1)
        outputs = 2 * self.gate(inputs)
        return outputs


class PPNet(torch.nn.Module):
    def __init__(self,
                 dense_in_features: int,
                 gate_in_features: int,
                 hidden_units: List[int],
                 activation: str = "torch.nn.ReLU",
                 use_bn: bool = False,
                 ):
        super().__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bn = use_bn
        self.output_dim = self.hidden_units[-1]

        self.gate_list = []
        self.mlp_list = []
        for i in range(len(hidden_units)):
            self.gate_list.append(
                GateNU(
                    dense_in_features=dense_in_features,
                    gate_in_features=gate_in_features,
                )
            )
            self.mlp_list.append(
                Perceptron(
                    in_features=dense_in_features if i == 0 else hidden_units[i - 1],
                    out_features=hidden_units[i],
                    activation=activation,
                    use_bn=use_bn,
                )
            )

    def forward(self, dense_inputs: torch.Tensor, gate_inputs: torch.Tensor) -> torch.Tensor:
        outputs = dense_inputs
        for i in range(len(self.hidden_units)):
            gate_outputs = self.gate_list[i](outputs, gate_inputs)
            mlp_inputs = torch.multiply(outputs, gate_outputs)
            outputs = self.mlp_list[i](mlp_inputs)
        return outputs


# TODO
class EPNet(torch.nn.Module):
    pass


# TODO
class PEPNet(torch.nn.Module):
    pass

