# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Any
import itertools
import json

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tdrec.protos.pipeline_pb2 import TrainConfig
from tdrec.utils import checkpoint_util
from tdrec.utils.logging_util import ProgressLogger, logger


def _log_train(
        step: int,
        losses: Dict[str, torch.Tensor],
        param_groups: List[Dict[str, Any]],
        plogger: ProgressLogger = None,
        summary_writer: SummaryWriter = None,
) -> None:
    """Logging current training step."""
    if plogger is not None:
        loss_strs = []
        lr_strs = []
        for k, v in losses.items():
            loss_strs.append(f"{k}:{v:.5f}")
        for i, g in enumerate(param_groups):
            lr_strs.append(f"lr_g{i}:{g['lr']:.5f}")
        total_loss = sum(losses.values())
        plogger.log(
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
          train_dataloader: DataLoader,
          train_config: TrainConfig,
          model_dir: str,
          ckpt_path: str = None,
          ):
    model.train()
    epoch_iter = range(train_config.num_epochs)

    plogger = ProgressLogger(desc="Training Model")
    summary_writer = SummaryWriter(model_dir)

    last_ckpt_step = 0
    i_step = 0
    losses = {}
    for i_epoch in epoch_iter:
        if i_step == 0 and ckpt_path is not None:
            last_ckpt_step = checkpoint_util.get_checkpoint_step(ckpt_path)
            checkpoint_util.restore_model(ckpt_path, model, optimizer)
        step_iter = itertools.count(i_step)
        train_iterator = iter(train_dataloader)
        for i_step in step_iter:
            try:
                batch = next(train_iterator)
            except StopIteration:
                break
            losses, _ = model(batch)
            if i_step % train_config.log_step_count_steps:
                _log_train(
                    i_step,
                    losses,
                    param_groups=optimizer.param_groups,
                    plogger=plogger,
                    summary_writer=summary_writer,
                )
            if i_step % train_config.save_checkpoints_steps:
                last_ckpt_step = i_step
                checkpoint_util.save_model(
                    os.path.join(model_dir, f"model.ckpt-{i_step + last_ckpt_step}"),
                    model,
                    optimizer,
                )
        logger.info(f"Finished Train Model Epoch {i_epoch + 1}.")
    _log_train(
        i_step,
        losses,
        param_groups=optimizer.param_groups,
        plogger=plogger,
        summary_writer=summary_writer,
    )
    checkpoint_util.save_model(
        os.path.join(model_dir, f"model.ckpt-{i_step + last_ckpt_step}"),
        model,
        optimizer,
    )
    if summary_writer is not None:
        summary_writer.close()


def evaluate_model(model: torch.nn.Module,
             eval_dataloader: DataLoader,
             eval_result_filename: str,
             global_step: int = None,
             global_epoch: int = None,
             summary_writer: SummaryWriter = None,
             ) -> Dict[str, torch.Tensor]:
    model.eval()
    _model = model.model
    eval_iterator = iter(eval_dataloader)
    step_iter = itertools.count(0)

    desc_suffix = ""
    if global_epoch:
        desc_suffix += f" Epoch-{global_epoch}"
    if global_step:
        desc_suffix += f" model-{global_step}"

    with torch.no_grad():
        for i_step in step_iter:
            try:
                batch = next(eval_iterator)
            except StopIteration:
                break
            _, (losses, predictions, batch) = model(batch)
            _model.update_metric(predictions, batch, losses)

    metric_result = _model.compute_metric()

    metric_str = " ".join([f"{k}:{v:0.6f}" for k, v in metric_result.items()])
    logger.info(f"Eval Result{desc_suffix}: {metric_str}")
    metric_result = {k: v.item() for k, v in metric_result.items()}
    if eval_result_filename:
        with open(eval_result_filename, "w") as f:
            json.dump(metric_result, f, indent=4)
    if summary_writer:
        for k, v in metric_result.items():
            summary_writer.add_scalar(f"metric/{k}", v, global_step or 0)

    return metric_result
