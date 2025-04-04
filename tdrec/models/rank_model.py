# -*- coding: utf-8 -*-

from typing import List, Dict
import datetime

import torch
import torchmetrics
from tdrec.protos.model_pb2 import ModelConfig
from tdrec.features.feature import BaseFeature
from tdrec.models.base_model import BaseModel
from tdrec.modules.mlp import MLP
from tdrec.utils.config_util import config_to_kwargs
from tdrec.metrics import GroupedAUC, COPC


class RankModel(BaseModel):
    def __init__(self,
                 model_config: ModelConfig,
                 features: List[BaseFeature],
                 labels: List[str],
                 sample_weight: str = None,
                 ):
        super().__init__(model_config, features, labels, sample_weight)
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initialize RankModel.")
        self._label_name = labels[0]

        self.top_mlp = MLP(in_features=model_config.backbone.output_dim, **config_to_kwargs(self._model_config.top_mlp))
        self.linear = torch.nn.Linear(self._model_config.top_mlp.hidden_units[-1], 1)

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = dict()
        backbone_output, _ = self._backbone(batch)
        output = self.top_mlp(backbone_output)
        output = self.linear(output)
        output = torch.squeeze(output, dim=1)
        predictions["probs"] = torch.sigmoid(output)
        return predictions

    def init_loss(self) -> None:
        loss_name = "ce_loss"
        self._loss_modules[loss_name] = torch.nn.BCELoss(
            reduction="none" if self._sample_weight else "mean",
        )

    def compute_loss(self, batch: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        probs = predictions["probs"]
        labels = batch[self._label_name].to(torch.float32)
        losses = {}
        loss_name = "ce_loss"
        losses[loss_name] = self._loss_modules[loss_name](probs, labels)
        if self._sample_weight:
            loss_weight = batch[self._sample_weight]
            losses[loss_name] = torch.mean(losses[loss_name] * loss_weight)
        return losses

    def init_metric(self) -> None:
        for metric_cfg in self._base_model_config.metrics:
            metric_type = metric_cfg.WhichOneof("metric")
            if metric_type == "auc":
                self._metric_modules[metric_type] = torchmetrics.AUROC(task="binary")
            elif metric_type == "grouped_auc":
                self._metric_modules[metric_type] = GroupedAUC()
            elif metric_type == "copc":
                self._metric_modules[metric_type] = COPC()
            else:
                raise ValueError(f"{metric_type} is not supported for this model.")

    def update_metric(self, batch: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        metrics = {}
        probs = predictions["probs"]
        labels = batch[self._label_name]
        for metric_cfg in self._base_model_config.metrics:
            metric_type = metric_cfg.WhichOneof("metric")
            oneof_metric_cfg = getattr(metric_cfg, metric_type)
            if metric_type == "auc":
                self._metric_modules[metric_type].update(probs, labels)
            elif metric_type == "grouped_auc":
                grouping_key = batch[oneof_metric_cfg.grouping_key]
                self._metric_modules[metric_type].update(probs, labels, grouping_key)
            elif metric_type == "copc":
                self._metric_modules[metric_type].update(probs, labels)
            else:
                raise ValueError(f"{metric_type} is not supported for this model.")
        return metrics
