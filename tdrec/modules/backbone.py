# -*- coding: utf-8 -*-

from typing import Dict, Any, Tuple
import datetime

import torch

from tdrec.utils.config_util import config_to_kwargs
from tdrec.features.feature_group import FeatureGroup
from tdrec.protos.backbone_pb2 import BackboneConfig, BlockConfig
from tdrec.modules.mlp import MLP
from tdrec.modules.din import DIN
from tdrec.modules.fm import FactorizationMachine
from tdrec.modules.dlrm import DLRM


class Backbone(torch.nn.Module):
    def __init__(self,
               config: BackboneConfig,
               feature_group_dict: Dict[str, FeatureGroup],
               ):
        super().__init__()
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initialize Backbone.")
        self._config = config
        self._feature_group_dict = feature_group_dict
        self._block_outputs = {}

        self._block_modules = torch.nn.ModuleDict()
        for block_config in self._config.blocks:
            module_type = block_config.WhichOneof("module")
            module_config = getattr(block_config, module_type)
            if module_type == "mlp":
                self._block_modules[block_config.name] = MLP(in_features=self._get_block_input_dim(block_config), **config_to_kwargs(module_config))
            elif module_type == "fm":
                self._block_modules[block_config.name] = FactorizationMachine()
            elif module_type == "dlrm":
                self._block_modules[block_config.name] = DLRM()
            elif module_type == "din":
                sequence_dim = self._get_block_input_dim(block_config)
                self._block_modules[block_config.name] = DIN(sequence_dim=sequence_dim, query_dim=sequence_dim, **config_to_kwargs(module_config))

            else:
                raise ValueError(
                    f"block[{block_config.name}] don't support [{module_type}] now."
                )
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] block_modules:{self._block_modules}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feature_group_values = dict()
        for k, v in self._feature_group_dict.items():
            feature_group_values[k] = v.build_group_input(batch)
        backbone_output = []
        for block in self._config.blocks:
            block_output = self.build_block_output(block, feature_group_values)
            backbone_output.append(block_output)
        backbone_output = torch.cat(backbone_output, dim=1)
        return backbone_output, self._block_outputs

    def _get_block_input_dim(self, block_config: BlockConfig):
        res = 0
        for input_name in block_config.feature_group_names:
            res += self._feature_group_dict[input_name].output_dim
        return res

    def build_block_output(self, block_config: BlockConfig, feature_group_values: Dict[str, Any]) -> torch.Tensor:
        block_inputs = []
        for input_name in block_config.feature_group_names:
            block_inputs.append(feature_group_values[input_name])
        module_type = block_config.WhichOneof("module")
        if module_type == "mlp":
            block_input = torch.cat(block_inputs, dim=1)
            block_output = self._block_modules[block_config.name](block_input)
        elif module_type in ["fm", "dlrm", "din"]:
            block_input = block_inputs[0]
            block_output = self._block_modules[block_config.name](block_input)
        else:
            raise ValueError(
                f"block[{block_config.name}] don't support [{module_type}] now."
            )
        self._block_outputs[block_config.name] = block_output
        return block_output
