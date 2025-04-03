# -*- coding: utf-8 -*-

from typing import List, Any, Dict
import datetime

import torch
import torchmetrics
from tdrec.protos.model_pb2 import ModelConfig
from tdrec.features.feature import BaseFeature
from tdrec.models.base_model import BaseModel
from tdrec.modules.mlp import MLP
from tdrec.modules.mmoe import MMoE
from tdrec.modules.dbmtl import DBMTL
from tdrec.utils.config_util import config_to_kwargs


class MultiTaskRankModel(BaseModel):
    def __init__(self,
                 model_config: ModelConfig,
                 features: List[BaseFeature],
                 labels: List[str],
                 ):
        super().__init__(model_config, features, labels)
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initialize MultiTaskRankModel.")

        self._task_tower_cfgs = list(self._model_config.task_towers)

        self._multi_task_model_type = self._model_config.WhichOneof("multi_task_model")
        self._multi_task_model_config = getattr(self._model_config, self._multi_task_model_type) if self._multi_task_model_type else None
        if self._multi_task_model_type == "mmoe":
            self._multi_task_model = MMoE(in_features=model_config.backbone.output_dim, **config_to_kwargs(self._multi_task_model_config))
        elif self._multi_task_model_type == "dbmtl":
            self._multi_task_model = DBMTL(in_features=model_config.backbone.output_dim, **config_to_kwargs(self._multi_task_model_config))
        elif self._multi_task_model_type == "esmm":
            pass
        else:
            self._multi_task_model = None

        self._task_tower = torch.nn.ModuleList()
        for task_tower in self.task_towers:
            self._task_tower.append(torch.nn.Sequential(
                MLP(in_features=task_tower.tower_input_dim, **config_to_kwargs(task_tower.mlp)),
                torch.nn.Linear(task_tower.mlp.hidden_units[-1], 1),
            ))

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = dict()
        backbone_output, _ = self._backbone(batch)
        task_input_list = self._multi_task_model(backbone_output)
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_output = self._task_tower[i](task_input_list[i])
            predictions[task_tower_cfg.label_name + "_probs"] = torch.sigmoid(tower_output)
        return predictions

    def init_loss(self) -> None:
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            self._loss_modules[task_tower_cfg.label_name + "bce_loss"] = torch.nn.BCELoss(
                reduction="mean",
            )
