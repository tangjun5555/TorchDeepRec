# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Any
import itertools
import json
import datetime
import time

import torch
from torch.utils.data import DataLoader

from tdrec.protos.pipeline_pb2 import TrainConfig
from tdrec.optimizer.lr_scheduler import BaseLR
from tdrec.utils import checkpoint_util
from tdrec.constant import Mode


def _log_train(
        step: int,
        cost_time: float,
        losses: Dict[str, torch.Tensor],
        param_groups: List[Dict[str, Any]],
) -> None:
    """
    Logging current training step.
    """
    loss_strs = []
    for k, v in losses.items():
        loss_strs.append(f"{k}:{v:.5f}")
    lr_str = f"lr:{param_groups[0]['lr']:.5f}"
    total_loss = sum(losses.values())
    print(
        f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training Model",
        f"step:{step}",
        f"cost_time:{round(cost_time, 1)}m",
        f"{lr_str}",
        f"{' '.join(loss_strs)} total_loss: {total_loss:.5f}",
    )


def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                lr_scheduler: BaseLR,
                train_dataloader: DataLoader,
                train_config: TrainConfig,
                model_dir: str,
                global_step: int,
                ) -> int:
    model.train()
    epoch_iter = range(train_config.num_epochs)

    i_step = global_step
    losses = {}
    start_time = time.time()
    for i_epoch in epoch_iter:
        step_iter = itertools.count(i_step + 1)
        train_iterator = iter(train_dataloader)
        for i_step in step_iter:
            try:
                batch = next(train_iterator)
            except StopIteration:
                i_step -= 1
                break

            optimizer.zero_grad()
            total_loss, (losses, _, _) = model(batch)
            total_loss.backward()
            optimizer.step()
            if not lr_scheduler.by_epoch:
                lr_scheduler.step()

            if i_step % train_config.log_step_count_steps == 0:
                current_time = time.time()
                cost_time = (current_time - start_time) / 60.0
                start_time = current_time
                _log_train(i_step, cost_time, losses, param_groups=optimizer.param_groups)

        if lr_scheduler.by_epoch:
            lr_scheduler.step()
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished Train Model Epoch {i_epoch + 1}.")

    cost_time = (time.time() - start_time) / 60.0
    _log_train(i_step, cost_time, losses, param_groups=optimizer.param_groups)
    checkpoint_util.save_model(
        os.path.join(model_dir, f"model.ckpt-{i_step}"),
        model,
        optimizer,
    )
    checkpoint_util.keep_checkpoint_max(model_dir, train_config.keep_checkpoint_max)
    return i_step


def evaluate_model(model: torch.nn.Module,
                   eval_dataloader: DataLoader,
                   model_dir: str,
                   eval_result_filename: str,
                   global_step: int,
                   mode=Mode.TRAIN,
                   ) -> Dict[str, torch.Tensor]:
    model.eval()
    _model = model.model
    eval_iterator = iter(eval_dataloader)
    step_iter = itertools.count(0)

    desc_suffix = f"model-{global_step}"

    with torch.no_grad():
        for i_step in step_iter:
            try:
                batch = next(eval_iterator)
            except StopIteration:
                i_step -= 1
                break
            _, (losses, predictions, _) = model(batch)
            _model.update_metric(batch, predictions)
    metric_result = _model.compute_metric()

    metric_result = {k: v.item() for k, v in metric_result.items()}
    for k, v in losses.items():
        metric_result[k] = v.item()
    metric_str = " ".join([f"{k}:{v:0.6f}" for k, v in metric_result.items()])
    print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mode} Eval Result[{desc_suffix}]: {metric_str}")
    with open(eval_result_filename, "w") as f:
        json.dump(metric_result, f, indent=4)
    return metric_result
