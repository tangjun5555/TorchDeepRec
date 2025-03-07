# -*- coding: utf-8 -*-

from typing import List

from tdrec.constant import Batch
from tdrec.protos.model_pb2 import FeatureGroupConfig
from tdrec.features.feature import BaseFeature


class FeatureGroup(object):
    def __int__(self,
                feature_group_config: FeatureGroupConfig,
                features: List[BaseFeature],
                ):
        self._config = feature_group_config
        self._features_dict = {base_feature.name:base_feature for base_feature in features}

    def build_group_input(self, batch: Batch):
        group_features = {}
        for name in self._config.feature_names:
            values = batch.features[name]
            group_features[name] = self._features_dict[name].to_dense(values)
        return group_features
