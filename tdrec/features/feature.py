# -*- coding: utf-8 -*-

import pyarrow as pa
from typing import List, Dict
from abc import abstractmethod

from tdrec.protos.feature_pb2 import FeatureConfig
from tdrec.datasets.dataset import ParsedData


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
            return self.config.input_names[0]

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


def create_features(feature_configs: List[FeatureConfig]):
    features = []
    for feature_config in feature_configs:
        config = getattr(feature_config, feature_config.WhichOneof("feature"))
    return features
