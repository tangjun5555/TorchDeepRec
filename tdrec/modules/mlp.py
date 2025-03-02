# -*- coding: utf-8 -*-

import torch
from typing import List
from tdrec.utils.load_class import load_by_path


class Perceptron(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: str = "torch.nn.ReLU",
                 use_bn: bool = False,
                 ):
        super().__init__()
        self.activation = activation
        self.use_bn = use_bn
        self.perceptron = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features, bias=False if use_bn else True)
        )

        if use_bn:
            self.perceptron.append(torch.nn.BatchNorm1d(out_features))
        if activation:
            act_module = load_by_path(activation)()
            if act_module:
                self.perceptron.append(act_module)
            else:
                raise ValueError(f"Unknown activation method: {activation}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs:
            torch.Tensor: shape is [B, in_features]
        :return:
            torch.Tensor: shape is [B, out_features]
        """
        return self.perceptron(inputs)


class MLP(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_units: List[int],
                 activation: str = "torch.nn.ReLU",
                 use_bn: bool = False,
                 ):
        super().__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bn = use_bn
        self.output_dim = self.hidden_units[-1]

        self.mlp = torch.nn.Sequential(
            *[
                Perceptron(
                    in_features=in_features if i == 0 else hidden_units[i - 1],
                    out_features=hidden_units[i],
                    activation=activation,
                    use_bn=use_bn,
                )
                for i in range(len(hidden_units))
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs:
            torch.Tensor: shape is [B, in_features]
        :return:
            torch.Tensor: shape is [B, output_dim]
        """
        return self.mlp(inputs)
