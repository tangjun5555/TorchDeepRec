# -*- coding: utf-8 -*-

from typing import Dict, Any

import torch
from tdrec.utils.config_util import config_to_kwargs
from tdrec.datasets.dataset import Batch
from tdrec.features.feature_group import FeatureGroup
from tdrec.protos.backbone_pb2 import BackboneConfig, BlockConfig
from tdrec.modules.mlp import MLP
from tdrec.modules.din import DIN


class Backbone(torch.nn.Module):
    def __init__(self,
               config: BackboneConfig,
               feature_group_dict: Dict[str, FeatureGroup],
               ):
        super().__init__()
        self._config = config
        self._feature_group_dict = feature_group_dict
        self._block_outputs = {}

    def forward(self, batch: Batch) -> torch.Tensor:
        feature_group_values = dict()
        for k, v in self._feature_group_dict.items():
            feature_group_values[k] = v.build_group_input(batch)
        backbone_output = []
        for block in self._config.blocks:
            block_output = self.build_block_output(block, feature_group_values)
            backbone_output.append(block_output)
        backbone_output = torch.cat(backbone_output, dim=1)
        return backbone_output

    def build_block_output(self, block_config: BlockConfig, feature_group_values: Dict[str, Any]) -> torch.Tensor:
        block_inputs = []
        for input_name in block_config.feature_group_names:
            block_inputs.append(feature_group_values[input_name])
        module_type = block_config.WhichOneof("module")
        module_config = getattr(block_config, module_type)
        if module_type == "mlp":
            block_input = torch.cat(block_inputs, dim=1)
            block_output = MLP(in_features=block_input.shape[1], **config_to_kwargs(module_config))(block_input)
        elif module_type == "din":
            block_input = block_inputs[0]
            sequence_dim = list(block_input.values())[0].shape[2]
            block_output = DIN(sequence_dim=sequence_dim, query_dim=sequence_dim, **config_to_kwargs(module_config))(block_input)
        else:
            raise ValueError(
                f"block[{block_config.name}] don't support [{module_type}] now."
            )
        self._block_outputs[block_config.name] = block_output
        return block_output
