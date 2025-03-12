# -*- coding: utf-8 -*-

from typing import Dict, Any

from tdrec.datasets.dataset import Batch
from tdrec.protos.backbone_pb2 import BackboneConfig, BlockConfig


class Backbone(object):
    def __init(self,
               config: BackboneConfig,
               feature_group_dict: Dict[str, Any],
               ):
        self._config = config
        self._feature_group_dict = feature_group_dict

        self._block_outputs = {}
        for block in config.blocks:
            if len(block.inputs) == 0:
                raise ValueError(f"block:{block.name} must takes at least one input: %s")
            module = block.WhichOneof("module")

    def build_block_output(self, block_config: BlockConfig):
        block_inputs = []
        for input_node in block_config.inputs:
            input_type = input_node.WhichOneof("name")
            input_name = getattr(input_node, input_type)
            if input_type == "feature_group_name":
                block_inputs.append(self._feature_group_dict[input_name])
            elif input_type == "block_name":
                block_inputs.append(self._block_outputs[input_name])
            else:
                raise ValueError(
                    f"block[{block_config.name}] don't support input[{input_type}] now."
                )

        module = block_config.module.WhichOneof("module")
        module_config = getattr(block_config, module)

        self._block_outputs[block_config.name] = None

    def build_backbone_output(self,  batch: Batch):
        pass
