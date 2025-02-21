# -*- coding: utf-8 -*-

import torch
from typing import List, Any, Dict

from tdrec.datasets.dataset import Batch
from tdrec.protos.model_pb2 import ModelConfig
from tdrec.features.feature import Feature


class BaseModel(torch.nn.Module):
    def __init__(self,
                 model_config: ModelConfig,
                 features: List[Feature],
                 labels: List[str],
                 **kwargs: Any,
                 ):
        super().__init__(**kwargs)
        self._base_model_config = model_config
        self._features = features
        self._labels = labels

        self._model_type = model_config.WhichOneof("model")
        self._model_config = getattr(model_config, self._model_type) if self._model_type else None

        self._metric_modules = torch.nn.ModuleDict()
        self._loss_modules = torch.nn.ModuleDict()

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        return self.predict(batch)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Predict the model.
        :param batch:
            input batch data.
        :return:
            a dict of predicted result.
        """
        raise NotImplementedError

    def compute_loss(self, batch: Batch, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss.
        :param batch:
            input batch data.
        :param predictions:
            a dict of predicted result.
        :return:
            a dict of loss result.
        """
        raise NotImplementedError

    def compute_metric(self) -> Dict[str, torch.Tensor]:
        """
        Calculate the metric.
        :return:
            a dict of metric result.
        """
        metric_results = {}
        for metric_name, metric in self._metric_modules.items():
            metric_results[metric_name] = metric.compute()
            metric.reset()
        return metric_results
