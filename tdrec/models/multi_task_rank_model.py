# -*- coding: utf-8 -*-

from typing import List, Dict
import datetime

import torch
import torchmetrics

from tdrec.protos.model_pb2 import ModelConfig
from tdrec.features.feature import BaseFeature
from tdrec.metrics import GroupedAUC, COPC
from tdrec.models.base_model import BaseModel
from tdrec.modules.mlp import MLP
from tdrec.modules.mmoe import MMoE
from tdrec.modules.dbmtl import DBMTL
from tdrec.modules.esmm import ESMM
from tdrec.utils.config_util import config_to_kwargs


class MultiTaskRankModel(BaseModel):
    def __init__(self,
                 model_config: ModelConfig,
                 features: List[BaseFeature],
                 labels: List[str],
                 sample_weight: str = None,
                 ):
        super().__init__(model_config, features, labels, sample_weight)
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initialize MultiTaskRankModel.")
        self._task_tower_cfgs = list(self._model_config.task_towers)
        self._num_task = len(self._task_tower_cfgs)

        self._multi_task_model_type = self._model_config.WhichOneof("multi_task_model")
        self._multi_task_model_config = getattr(self._model_config, self._multi_task_model_type) if self._multi_task_model_type else None
        if self._multi_task_model_type == "mmoe":
            self._multi_task_model = MMoE(in_features=model_config.backbone.output_dim, num_task=self._num_task, **config_to_kwargs(self._multi_task_model_config))
        elif self._multi_task_model_type == "dbmtl":
            self._multi_task_model = DBMTL(in_features=model_config.backbone.output_dim, **config_to_kwargs(self._multi_task_model_config))
        elif self._multi_task_model_type == "esmm":
            self._multi_task_model = ESMM(in_features=model_config.backbone.output_dim, num_task=self._num_task, **config_to_kwargs(self._multi_task_model_config))
        else:
            raise ValueError(f"{self._multi_task_model_type} is not supported.")

        self._task_tower = torch.nn.ModuleList()
        for task_tower in self.task_towers:
            self._task_tower.append(torch.nn.Sequential(
                MLP(in_features=task_tower.tower_input_dim if task_tower.tower_input_dim else self._multi_task_model.output_dim, **config_to_kwargs(task_tower.mlp)),
                torch.nn.Linear(task_tower.mlp.hidden_units[-1], 1),
            ))

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = dict()
        backbone_output, _ = self._backbone(batch)
        task_input_list = self._multi_task_model(backbone_output)
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_output = self._task_tower[i](task_input_list[i])
            predictions[task_tower_cfg.label_name + "_" +  "probs"] = torch.sigmoid(tower_output)
        return predictions

    def init_loss(self) -> None:
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            loss_name = task_tower_cfg.label_name + "_" + "ce_loss"
            self._loss_modules[loss_name] = torch.nn.BCELoss(reduction="mean")

    def compute_loss(self, batch: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        pre_probs = None
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            probs = predictions[task_tower_cfg.label_name + "_" +  "probs"]
            labels = batch[task_tower_cfg.label_name].to(torch.float32)
            loss_name = task_tower_cfg.label_name + "_" + "ce_loss"
            if self._multi_task_model_type == "esmm" and i > 0:
                probs = probs * pre_probs
            losses[loss_name] = self._loss_modules[loss_name](probs, labels) * task_tower_cfg.weight
            pre_probs = probs
        return losses

    def init_metric(self) -> None:
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            for metric_cfg in self._base_model_config.metrics:
                metric_type = metric_cfg.WhichOneof("metric")
                metric_name = task_tower_cfg.label_name + "_" + metric_type
                if metric_type == "auc":
                    self._metric_modules[metric_name] = torchmetrics.AUROC(task="binary")
                elif metric_type == "grouped_auc":
                    self._metric_modules[metric_name] = GroupedAUC()
                elif metric_type == "copc":
                    self._metric_modules[metric_name] = COPC()
                else:
                    raise ValueError(f"{metric_type} is not supported for this model.")

    def update_metric(self, batch: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        metrics = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            probs = predictions[task_tower_cfg.label_name + "_" + "probs"]
            labels = batch[task_tower_cfg.label_name]
            for metric_cfg in self._base_model_config.metrics:
                metric_type = metric_cfg.WhichOneof("metric")
                metric_name = task_tower_cfg.label_name + "_" + metric_type
                oneof_metric_cfg = getattr(metric_cfg, metric_type)
                if metric_type == "auc":
                    self._metric_modules[metric_name].update(probs, labels)
                elif metric_type == "grouped_auc":
                    grouping_key = batch[oneof_metric_cfg.grouping_key]
                    self._metric_modules[metric_name].update(probs, labels, grouping_key)
                elif metric_type == "copc":
                    self._metric_modules[metric_name].update(probs, labels)
                else:
                    raise ValueError(f"{metric_type} is not supported for this model.")
        return metrics
