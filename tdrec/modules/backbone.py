# -*- coding: utf-8 -*-

from typing import List, Dict, Any

from tdrec.protos.backbone_pb2 import BackboneConfig, BlockConfig
from tdrec.features.feature import BaseFeature


class Backbone(object):
    def __init(self,
               config: BackboneConfig,
               features: List[BaseFeature],
               ):
        self._config = config
        self._features = features

        self._feature_group_inputs = {}
        self._block_outputs = {}

        for block in config.blocks:
            if len(block.inputs) == 0:
                raise ValueError(f"block:{block.name} must takes at least one input: %s")
            module = block.WhichOneof("module")

    def block_input(self, config: BlockConfig, block_outputs: Dict[str, Any]):
        inputs = []
        for input_node in config.inputs:
            input_type = input_node.WhichOneof("name")
            input_name = getattr(input_node, input_type)
            if input_type == "feature_group_name":
                input_feature = block_outputs[input_name]
            else:
                pass
