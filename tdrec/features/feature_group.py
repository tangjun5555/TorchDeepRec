# -*- coding: utf-8 -*-

from typing import List, Dict

import torch
from tdrec.protos.model_pb2 import FeatureGroupConfig, FeatureGroupType
from tdrec.features.feature import BaseFeature


class FeatureGroup(object):
    def __init__(self,
                feature_group_config: FeatureGroupConfig,
                features: List[BaseFeature],
                ):
        self._config = feature_group_config
        self._features_dict = {base_feature.name:base_feature for base_feature in features}

    def output_dim(self) -> int:
        res = 0
        if self._config.group_type == FeatureGroupType.Deep:
            for name in self._config.feature_names:
                res += self._features_dict[name].output_dim()
        elif self._config.group_type == FeatureGroupType.Sequence_Attention:
            for name in self._config.feature_names:
                res += self._features_dict[name].output_dim()
                break
        else:
            raise ValueError(
                f"feature_group[{self._config.group_name}] don't support [{self._config.group_type}] now."
            )
        return res

    def build_group_input(self, batch: Dict[str, torch.Tensor]):
        if self._config.group_type == FeatureGroupType.Deep:
            return self.build_dense_group_input(batch)
        elif self._config.group_type == FeatureGroupType.Sequence_Attention:
            return self.build_sequence_attention_group_input(batch)
        else:
            raise ValueError(
                f"feature_group[{self._config.group_name}] don't support [{self._config.group_type}] now."
            )

    def build_dense_group_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        group_features = []
        for name in self._config.feature_names:
            values = batch[self._features_dict[name].input_name]
            values = self._features_dict[name].to_dense(values)
            group_features.append(values)
        group_features = torch.cat(group_features, dim=1)
        return group_features

    def build_sequence_attention_group_input(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        group_features = {}
        query_name = f"{self._config.group_name}.query"
        sequence_name = f"{self._config.group_name}.sequence"
        sequence_length_name = f"{self._config.group_name}.sequence_length"

        query_features = []
        sequence_features = []
        for name in self._config.feature_names:
            values = batch[self._features_dict[name].input_name]
            if len(values.shape) == 1:
                values = self._features_dict[name].to_dense(values)
                query_features.append(values)
            else:
                sequence_length = torch.count_nonzero(values, dim=1)
                group_features[sequence_length_name] = sequence_length
                values = self._features_dict[name].to_dense(values)
                sequence_features.append(values)
        group_features[query_name] = torch.cat(query_features, dim=1)
        group_features[sequence_name] = torch.cat(sequence_features, dim=1)
        return group_features
