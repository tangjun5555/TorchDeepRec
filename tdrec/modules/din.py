# -*- coding: utf-8 -*-

from typing import Dict, Any
import datetime

import torch
from tdrec.modules.mlp import MLP


@torch.fx.wrap
def _arange(end: int, device: torch.device) -> torch.Tensor:
    return torch.arange(end, device=device)


class DIN(torch.nn.Module):
    def __init__(self,
                 sequence_dim: int,
                 query_dim: int,
                 feature_group: str,
                 attn_mlp: Dict[str, Any],
                 ) -> None:
        super().__init__()
        self._query_dim = query_dim
        self._sequence_dim = sequence_dim
        assert query_dim == sequence_dim

        self.mlp = MLP(in_features=sequence_dim * 4, **attn_mlp)
        self.linear = torch.nn.Linear(self.mlp.output_dim, 1)

        self._query_name = f"{feature_group}.query"
        self._sequence_name = f"{feature_group}.sequence"
        self._sequence_length_name = f"{feature_group}.sequence_length"

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        query = inputs[self._query_name]
        sequence = inputs[self._sequence_name]
        sequence_length = inputs[self._sequence_length_name]
        max_seq_length = sequence.size(1)
        sequence_mask = _arange(
            max_seq_length, device=sequence_length.device
        ).unsqueeze(0) < sequence_length.unsqueeze(1)

        queries = query.unsqueeze(1).expand(-1, max_seq_length, -1)

        attn_input = torch.cat(
            [queries, sequence, queries - sequence, queries * sequence],
            dim=-1,
        )
        attn_output = self.mlp(attn_input)
        attn_output = self.linear(attn_output)
        attn_output = attn_output.transpose(1, 2)

        padding = torch.ones_like(attn_output) * (-(2 ** 32) + 1)
        scores = torch.where(sequence_mask.unsqueeze(1), attn_output, padding)
        scores = torch.softmax(scores, dim=-1)
        outputs = torch.matmul(scores, sequence).squeeze(1)
        f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] DIN outputs.size:{outputs.size()}."
        return outputs
