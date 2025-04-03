# -*- coding: utf-8 -*-

import torch
from abc import abstractmethod
from typing import List, Dict, Optional

from tdrec.protos.model_pb2 import ModelConfig
from tdrec.features.feature import BaseFeature
from tdrec.features.feature_group import FeatureGroup
from tdrec.modules.backbone import Backbone
from tdrec.utils.load_class import get_register_class_meta

_MODEL_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_MODEL_CLASS_MAP)


class BaseModel(torch.nn.Module, metaclass=_meta_cls):
    def __init__(self,
                 model_config: ModelConfig,
                 features: List[BaseFeature],
                 labels: List[str],
                 sample_weight: str = None,
                 ):
        super().__init__()
        self._base_model_config = model_config
        self._features = torch.nn.ModuleList(features)
        self._labels = labels
        self._sample_weight = sample_weight

        self._model_type = model_config.WhichOneof("model")
        self._model_config = getattr(model_config, self._model_type) if self._model_type else None

        self._metric_modules = torch.nn.ModuleDict()
        self._loss_modules = torch.nn.ModuleDict()

        feature_group_dict = dict()
        for feature_group in model_config.feature_groups:
            feature_group_dict[feature_group.group_name] = FeatureGroup(feature_group, features)
        self._feature_group_dict = feature_group_dict

        self._backbone = Backbone(model_config.backbone, feature_group_dict)

        self._input_feature_names = sorted(list(set([feature.config.input_name for feature in features])))
        print(f"input_feature_names:{self._input_feature_names}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.predict(batch)

    @abstractmethod
    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict the model.
        :param batch:
            input batch data.
        :return:
            a dict of predicted result.
        """
        raise NotImplementedError

    @abstractmethod
    def init_loss(self) -> None:
        """
        Initialize loss modules.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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

    @abstractmethod
    def init_metric(self) -> None:
        """
        Initialize metric modules.
        """
        raise NotImplementedError

    @abstractmethod
    def update_metric(self, batch: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate the metric.
        :return:
            a dict of metric result.
        """
        raise NotImplementedError

    def compute_metric(self) -> Dict[str, torch.Tensor]:
        """Compute metric.

        Return:
            metric_result (dict): a dict of metric result tensor.
        """
        metric_results = {}
        for metric_name, metric in self._metric_modules.items():
            metric_results[metric_name] = metric.compute()
            metric.reset()
        return metric_results


def create_model(model_config: ModelConfig,
                 features: List[BaseFeature],
                 labels: List[str],
                 sample_weight: Optional[str] = None,
                ) -> BaseModel:
    model_type = model_config.WhichOneof("model")
    model_cls_name = getattr(model_config, model_type).__class__.__name__
    model_cls = BaseModel.create_class(model_cls_name)
    model = model_cls(model_config, features, labels, sample_weight)
    return model


class TrainWrapper(torch.nn.Module):
    """Model train wrapper for pipeline."""
    def __init__(self, model: BaseModel) -> None:
        super().__init__()
        self.model = model
        self.model.init_loss()
        self.model.init_metric()

    def forward(self, inputs: Dict[str, torch.Tensor]):
        batch = inputs
        predictions = self.model.predict(batch)
        losses = self.model.compute_loss(batch, predictions)
        total_loss = torch.stack(list(losses.values())).sum()

        losses = {k: v.detach() for k, v in losses.items()}
        predictions = {k: v.detach() for k, v in predictions.items()}
        return total_loss, (losses, predictions, batch)
