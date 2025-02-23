# -*- coding: utf-8 -*-

import torch
import unittest


class CommonTest(unittest.TestCase):
    def test_feature_manager(self):
        from tdrec.protos.feature_pb2 import FeatureManager, IdFeature

        feature_manager = FeatureManager()
        print(feature_manager.features, type(list(feature_manager.features)))

        id_feature = IdFeature()
        id_feature.input_names.append("uid")
        id_feature.feature_name = "uid--"

        print(id_feature.input_names)
        print(id_feature.input_names[0])
