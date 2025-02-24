# -*- coding: utf-8 -*-

import torch
from typing import List, Any, Dict

from tdrec.datasets.dataset import Batch
from tdrec.protos.model_pb2 import ModelConfig
from tdrec.features.feature import BaseFeature
from tdrec.models.base_model import BaseModel


class RankModel(BaseModel):
    def __init__(self,
                 model_config: ModelConfig,
                 features: List[BaseFeature],
                 labels: List[str],
                 **kwargs: Any,
                 ):
        super().__init__(model_config, features, labels, **kwargs)

