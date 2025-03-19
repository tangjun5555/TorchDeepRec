# -*- coding: utf-8 -*-

import os
import glob
from typing import Tuple, Optional

import torch
from tdrec.utils.logging_util import logger
from torch.distributed.checkpoint import (
    load,
    save,
)


def save_model(checkpoint_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    save(model.state_dict(), checkpoint_id=os.path.join(checkpoint_dir, "model"))
    if optimizer:
        save(
            optimizer.state_dict(),
            checkpoint_id=os.path.join(checkpoint_dir, "optimizer"),
        )
    logger.info(f"Successfully saved checkpoint to {checkpoint_dir}.")


def restore_model(checkpoint_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    if not checkpoint_dir:
        return
    model_ckpt_path = os.path.join(checkpoint_dir, "model")
    optim_ckpt_path = os.path.join(checkpoint_dir, "optimizer")
    if os.path.exists(model_ckpt_path):
        state_dict = model.state_dict()
        load(
            state_dict,
            checkpoint_id=model_ckpt_path,
        )
        model.load_state_dict(state_dict)
        logger.info(f"Successfully restored model state from {checkpoint_dir}.")
    if optimizer and os.path.exists(optim_ckpt_path):
        state_dict = optimizer.state_dict()
        load(
            state_dict,
            checkpoint_id=optim_ckpt_path,
        )
        optimizer.load_state_dict(state_dict)
        logger.info(f"Successfully restored optimizer state from {checkpoint_dir}.")
    logger.info(f"Successfully restored checkpoint from {checkpoint_dir}.")


def get_checkpoint_step(ckpt_path: str) -> int:
    """
    Get checkpoint step from ckpt_path.
    Args:
        ckpt_path: checkpoint path, such as xx/model.ckpt-2000.
    Return:
        ckpt_step: checkpoint step, such as 2000.
    """
    _, ckpt_name = os.path.split(ckpt_path)
    ckpt_name, ext = os.path.splitext(ckpt_name)
    if ext.startswith(".ckpt-"):
        ckpt_name = ext
    toks = ckpt_name.split("-")
    try:
        ckpt_step = int(toks[-1])
    except Exception:
        ckpt_step = 0
    return ckpt_step


def latest_checkpoint(model_dir: str) -> Tuple[Optional[str], int]:
    """
    Find latest checkpoint under a directory.
    Args:
        model_dir: model directory.
    Return:
        latest_ckpt_path: latest checkpoint path.
        latest_step: step of the latest checkpoint.
    """
    if not os.path.exists(model_dir):
        return None, 0
    ckpt_metas = glob.glob(os.path.join(model_dir, "model.ckpt-*"))
    if not ckpt_metas:
        return None, 0
    ckpt_metas.sort(key=lambda x: get_checkpoint_step(x))
    latest_ckpt_path = ckpt_metas[-1]
    return latest_ckpt_path, get_checkpoint_step(latest_ckpt_path)
