# -*- coding: utf-8 -*-

import torch

class FactorizationMachine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs:
            torch.Tensor: shape is [B, N, D]
        :return:
            torch.Tensor: shape is [B, D]
        """
        square_of_sum = torch.pow(torch.sum(inputs, dim=1), 2)
        sum_of_square = torch.sum(torch.pow(inputs, 2), dim=1)
        outputs = 0.5 * (square_of_sum - sum_of_square)
        return outputs
