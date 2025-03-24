# -*- coding: utf-8 -*-

import datetime

import torch


class DLRM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: torch.Tensor: shape is [B, N, D]
        :return:
            torch.Tensor: shape is [B, D]
        """
        # perform a dot product
        Z = torch.bmm(inputs, torch.transpose(inputs, 1, 2))
        _, ni, nj = Z.shape
        offset = 0
        li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
        lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
        outputs = Z[:, li, lj]
        f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DLRM outputs.size:{outputs.size()}."
        return outputs
