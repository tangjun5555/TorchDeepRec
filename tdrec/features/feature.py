# -*- coding: utf-8 -*-

import pyarrow as pa
from typing import List, Dict
from abc import abstractmethod

from tdrec.protos.feature_pb2 import FeatureConfig
from tdrec.datasets.dataset import ParsedData
from tdrec.features.id_feature import IdFeature
from tdrec.features.raw_feature import RawFeature


class BaseFeature(object):
    def __init__(self,
                 feature_config: FeatureConfig,
                 ):
        fc_type = feature_config.WhichOneof("feature")
        self._feature_config = feature_config
        self.config = getattr(self._feature_config, fc_type)

    @property
    def name(self) -> str:
        """
        Feature name.
        """
        if self.config.feature_name:
            return self.config.feature_name
        else:
            return self.config.input_name

    @abstractmethod
    def parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        """
        Parse input data for the feature impl.
        Args:
            input_data (dict): raw input feature data.
        Return:
            parsed feature data.
        """
        raise NotImplementedError


def create_features(feature_configs: List[FeatureConfig]) -> List[BaseFeature]:
    features = []
    for feature_config in feature_configs:
        config = getattr(feature_config, feature_config.WhichOneof("feature"))
        feat_cls_name = config.__class__.__name__
        if feat_cls_name == "IdFeature":
            feature = IdFeature(
                feature_config=feature_config,
            )
        elif feat_cls_name == "RawFeature":
            feature = RawFeature(
                feature_config=feature_config,
            )
        else:
            raise ValueError(
                f"feature type:{feat_cls_name} is not supported now."
            )
        features.append(feature)
    return features
