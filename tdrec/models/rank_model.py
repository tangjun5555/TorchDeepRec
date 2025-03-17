# -*- coding: utf-8 -*-

from typing import List, Any, Dict

import torch
import torchmetrics
from tdrec.datasets.dataset import Batch
from tdrec.protos.model_pb2 import ModelConfig
from tdrec.features.feature import BaseFeature
from tdrec.models.base_model import BaseModel
from tdrec.modules.mlp import MLP
from tdrec.utils.config_util import config_to_kwargs
from tdrec.metrics.grouped_auc import GroupedAUC


class RankModel(BaseModel):
    def __init__(self,
                 model_config: ModelConfig,
                 features: List[BaseFeature],
                 labels: List[str],
                 sample_weight: str = None,
                 **kwargs: Any,
                 ):
        super().__init__(model_config, features, labels, sample_weight, **kwargs)
        self._num_class = 1
        self._label_name = labels[0]
        self._sample_weight_name = sample_weight

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        predictions = dict()
        backbone_output = self.build_backbone_network(batch)
        output = MLP(in_features=backbone_output.shape[1], **self._model_config.top_mlp)(backbone_output)
        output = torch.nn.Linear(self._model_config.top_mlp.hidden_units[-1], 1)(output)
        output = torch.squeeze(output, dim=1)
        predictions["logits"] = output
        predictions["probs"] = torch.sigmoid(output)
        return predictions

    def init_loss(self) -> None:
        self._loss_modules["binary_cross_entropy"] = torch.nn.BCEWithLogitsLoss(
            reduction="none" if self._sample_weight_name else "mean",
        )

    def compute_loss(self, batch: Batch, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        pred = predictions["logits"]
        label = batch.labels[self._label_name]

        label = label.to(torch.float32)
        loss_name = "binary_cross_entropy"
        losses[loss_name] = self._loss_modules[loss_name](pred, label)
        if self._sample_weight_name:
            loss_weight = batch.sample_weight
            losses[loss_name] = torch.mean(losses[loss_name] * loss_weight)
        return losses

    def init_metric(self) -> None:
        for metric_cfg in self._base_model_config.metrics:
            metric_type = metric_cfg.WhichOneof("metric")
            oneof_metric_cfg = getattr(metric_cfg, metric_type)
            metric_kwargs = config_to_kwargs(oneof_metric_cfg)
            metric_name = metric_type
            if metric_type == "auc":
                assert self._num_class <= 2, (
                    f"num_class must less than 2 when metric type is {metric_type}"
                )
                self._metric_modules[metric_name] = torchmetrics.AUROC(
                    task="binary",
                    **metric_kwargs,
                )
            elif metric_type == "grouped_auc":
                assert self._num_class <= 2, (
                    f"num_class must less than 2 when metric type is {metric_type}"
                )
                self._metric_modules[metric_name] = GroupedAUC()
            else:
                raise ValueError(f"{metric_type} is not supported for this model")

    def update_metric(self, batch: Batch, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        metrics = {}
        label = batch.labels[self._label_name]
        for metric_cfg in self._base_model_config.metrics:
            metric_type = metric_cfg.WhichOneof("metric")
            oneof_metric_cfg = getattr(metric_cfg, metric_type)
            metric_name = metric_type
            if metric_type == "auc":
                pred = predictions["probs"]
                self._metric_modules[metric_name].update(pred, label)
            elif metric_type == "grouped_auc":
                pred = predictions["probs"]
                grouping_key = batch.features[oneof_metric_cfg.grouping_key]
                self._metric_modules[metric_name].update(pred, label, grouping_key)
            else:
                raise ValueError(f"{metric_type} is not supported for this model")
        return metrics
