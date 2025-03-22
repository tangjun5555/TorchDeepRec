# -*- coding: utf-8 -*-

import pyarrow as pa
from typing import List, Dict
from abc import abstractmethod

import torch
from tdrec.protos.feature_pb2 import FeatureUnit
from tdrec.constant import ParsedData
from tdrec.utils.load_class import get_register_class_meta

_FEATURE_CLASS_MAP = {}
_feature_meta_cls = get_register_class_meta(_FEATURE_CLASS_MAP)


class BaseFeature(torch.nn.Module, metaclass=_feature_meta_cls):
    def __init__(self,
                 feature_config: FeatureUnit,
                 ):
        super().__init__()
        fc_type = feature_config.WhichOneof("feature")
        self._feature_config = feature_config
        self.config = getattr(self._feature_config, fc_type)

    def forward(self, parsed_value: torch.Tensor) -> torch.Tensor:
        return self.to_dense(parsed_value)

    @property
    def name(self) -> str:
        """
        Feature name.
        """
        if self.config.feature_name:
            return self.config.feature_name
        else:
            return self.config.input_name

    @property
    def input_name(self) -> str:
        return self.config.input_name

    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

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

    @abstractmethod
    def to_dense(self, parsed_value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def create_features(feature_configs: List[FeatureUnit]) -> List[BaseFeature]:
    features = []
    for feature_config in feature_configs:
        oneof_feat_config = getattr(feature_config, feature_config.WhichOneof("feature"))
        feat_cls_name = oneof_feat_config.__class__.__name__
        feature = BaseFeature.create_class(feat_cls_name)(
            feature_config=feature_config,
        )
        features.append(feature)
    return features
