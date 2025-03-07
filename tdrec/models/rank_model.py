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
                 sample_weight: str = None,
                 **kwargs: Any,
                 ):
        super().__init__(model_config, features, labels, sample_weight, **kwargs)

        self._num_class = 2
        self._label_name = labels[0]

    def build_input(self, batch: Batch) -> Dict[str, torch.Tensor]:
        pass
