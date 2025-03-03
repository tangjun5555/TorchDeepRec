# -*- coding: utf-8 -*-

from tdrec.datasets.dataset import Batch
from tdrec.protos.model_pb2 import FeatureGroupConfig


class FeatureGroup(object):
    def __int__(self,
                feature_group_config: FeatureGroupConfig,
                ):
        self._config = feature_group_config


    def build_group_input(self, batch: Batch):
        group_features = {}



        return group_features

