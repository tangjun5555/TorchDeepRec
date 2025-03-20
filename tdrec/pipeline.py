# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Any
import itertools
import json
import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tdrec.protos.pipeline_pb2 import TrainConfig
from tdrec.optimizer.lr_scheduler import BaseLR
from tdrec.utils import checkpoint_util


def _log_train(
        step: int,
        losses: Dict[str, torch.Tensor],
        param_groups: List[Dict[str, Any]],
        summary_writer: SummaryWriter = None,
) -> None:
    """
    Logging current training step.
    """
    loss_strs = []
    lr_strs = []
    for k, v in losses.items():
        loss_strs.append(f"{k}:{v:.5f}")
    for i, g in enumerate(param_groups):
        lr_strs.append(f"lr_g{i}:{g['lr']:.5f}")
    total_loss = sum(losses.values())
    print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training Model-",
        step,
        f"{' '.join(lr_strs)} {' '.join(loss_strs)} total_loss: {total_loss:.5f}",
    )
    if summary_writer is not None:
        total_loss = sum(losses.values())
        for k, v in losses.items():
            summary_writer.add_scalar(f"loss/{k}", v, step)
        summary_writer.add_scalar("loss/total", total_loss, step)
        for i, g in enumerate(param_groups):
            summary_writer.add_scalar(f"lr/g{i}", g["lr"], step)


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

    summary_writer = SummaryWriter(model_dir)

    i_step = global_step
    losses = {}
    for i_epoch in epoch_iter:
        step_iter = itertools.count(i_step)
        train_iterator = iter(train_dataloader)
        for i_step in step_iter:
            try:
                batch = next(train_iterator)
            except StopIteration:
                i_step -= 1
                break
            total_loss, (losses, _, _) = model(batch)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if not lr_scheduler.by_epoch:
                lr_scheduler.step()

            if i_step % train_config.log_step_count_steps == 0:
                _log_train(i_step, losses, param_groups=optimizer.param_groups, summary_writer=summary_writer)
        if lr_scheduler.by_epoch:
            lr_scheduler.step()
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finished Train Model Epoch {i_epoch + 1}.")
    _log_train(i_step, losses, param_groups=optimizer.param_groups, summary_writer=summary_writer)
    checkpoint_util.save_model(
        os.path.join(model_dir, f"model.ckpt-{i_step}"),
        model,
        optimizer,
    )
    if summary_writer is not None:
        summary_writer.close()
    return i_step


def evaluate_model(model: torch.nn.Module,
                   eval_dataloader: DataLoader,
                   model_dir: str,
                   eval_result_filename: str,
                   global_step: int,
                   ) -> Dict[str, torch.Tensor]:
    model.eval()
    _model = model.model
    eval_iterator = iter(eval_dataloader)
    step_iter = itertools.count(0)

    summary_writer = SummaryWriter(os.path.join(model_dir, "eval"))

    desc_suffix = ""
    if global_step:
        desc_suffix += f" model-{global_step}"

    with torch.no_grad():
        for i_step in step_iter:
            try:
                batch = next(eval_iterator)
            except StopIteration:
                i_step -= 1
                break
            _, (_, predictions, _) = model(batch)
            _model.update_metric(batch, predictions)
    metric_result = _model.compute_metric()

    metric_str = " ".join([f"{k}:{v:0.6f}" for k, v in metric_result.items()])
    print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Eval Result{desc_suffix}: {metric_str}")
    metric_result = {k: v.item() for k, v in metric_result.items()}
    if eval_result_filename:
        with open(eval_result_filename, "w") as f:
            json.dump(metric_result, f, indent=4)
    if summary_writer:
        for k, v in metric_result.items():
            summary_writer.add_scalar(f"metric/{k}", v, global_step or 0)
    return metric_result
