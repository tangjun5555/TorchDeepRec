# -*- coding: utf-8 -*-

from tdrec.protos.feature_pb2 import FeatureConfig


class Feature(object):
    def __init__(self,
                 feature_config: FeatureConfig,
                 ):
        fc_type = feature_config.WhichOneof("feature")
        self._feature_config = feature_config
        self.config = getattr(self._feature_config, fc_type)
