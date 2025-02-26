# -*- coding: utf-8 -*-

import torch
from typing import List, Any, Dict, Optional

from tdrec.datasets.dataset import Batch
from tdrec.datasets.data_parser import DataParser
from tdrec.protos.model_pb2 import ModelConfig
from tdrec.features.feature import BaseFeature


class BaseModel(torch.nn.Module):
    def __init__(self,
                 model_config: ModelConfig,
                 features: List[BaseFeature],
                 labels: List[str],
                 sample_weight: str = None,
                 **kwargs: Any,
                 ):
        super().__init__(**kwargs)
        self._base_model_config = model_config
        self._features = features
        self._labels = labels
        self._sample_weight = sample_weight

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

    def init_loss(self) -> None:
        """
        Initialize loss modules.
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

    def init_metric(self) -> None:
        """
        Initialize metric modules.
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

    def build_group_input(self, batch: Batch, group_name: str):
        group_features = {}



        return group_features


def create_model(model_config: ModelConfig,
                 features: List[BaseFeature],
                 labels: List[str],
                 sample_weight: Optional[str] = None,
                ) -> BaseModel:
    model_type = model_config.WhichOneof("model")
    # TODO
    base_model: BaseModel = None
    if model_type == "RankModel":
        pass
    elif model_type == "MultiTaskRankModel":
        pass
    else:
        raise ValueError(
            f"model type:{model_type} is not supported now."
        )
    return base_model


class TrainWrapper(torch.nn.Module):
    def __init__(self, model: BaseModel) -> None:
        super().__init__()
        self.model = model
        self.model.init_loss()
        self.model.init_metric()

    def forward(self, batch: Batch):
        predictions = self.model.predict(batch)
        losses = self.model.compute_loss(batch, predictions)
        total_loss = torch.stack(list(losses.values())).sum()

        losses = {k: v.detach() for k, v in losses.items()}
        predictions = {k: v.detach() for k, v in predictions.items()}
        return total_loss, (losses, predictions, batch)


class ScriptWrapper(torch.nn.Module):
    def __init__(self, model: BaseModel) -> None:
        super().__init__()
        self.model = model
        self._features = self.model._features
        self._data_parser = DataParser(self._features)

    def get_batch(self, data: Dict[str, torch.Tensor], device: torch.device = "cpu") -> Batch:
        """
        Get batch.
        """
        batch = self._data_parser.to_batch(data)
        batch = batch.to(device, non_blocking=True)
        return batch
