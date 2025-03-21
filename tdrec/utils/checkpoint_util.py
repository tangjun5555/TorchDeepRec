# -*- coding: utf-8 -*-

import os
import glob
from typing import Tuple, Optional
import datetime
import shutil

import torch


def save_model(checkpoint_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] create dir {checkpoint_dir}")
    model_ckpt_path = os.path.join(checkpoint_dir, "model")
    optim_ckpt_path = os.path.join(checkpoint_dir, "optimizer")
    torch.save(model.state_dict(), model_ckpt_path)
    # 打印模型的状态字典
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully saved model checkpoint to {checkpoint_dir}.")
    if optimizer:
        torch.save(optimizer.state_dict(), optim_ckpt_path)
        # 打印优化器的状态字典
        print("Optimizer's state_dict:")
        # for var_name in optimizer.state_dict():
        #     print(var_name, "\t", optimizer.state_dict()[var_name])
        var_name = "param_groups"
        print(var_name, "\t", optimizer.state_dict()[var_name])
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully saved optimizer checkpoint to {checkpoint_dir}.")


def restore_model(checkpoint_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    if not checkpoint_dir:
        return
    model_ckpt_path = os.path.join(checkpoint_dir, "model")
    optim_ckpt_path = os.path.join(checkpoint_dir, "optimizer")
    if os.path.exists(model_ckpt_path):
        ckpt = torch.load(model_ckpt_path, weights_only=True)
        model.load_state_dict(ckpt)
        model.eval()
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully restored model state from {checkpoint_dir}.")
    if optimizer and os.path.exists(optim_ckpt_path):
        ckpt = torch.load(optim_ckpt_path, weights_only=True)
        optimizer.load_state_dict(ckpt)
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully restored optimizer state from {checkpoint_dir}.")


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


def keep_checkpoint_max(model_dir: str, max_version: int = 10):
    if not os.path.exists(model_dir):
        return
    if max_version <= 0:
        return
    ckpt_metas = glob.glob(os.path.join(model_dir, "model.ckpt-*"))
    if not ckpt_metas:
        return
    if len(ckpt_metas) <= max_version:
        return
    ckpt_metas.sort(key=lambda x: get_checkpoint_step(x))
    while len(ckpt_metas) > max_version:
        old_ckpt = ckpt_metas.pop(0)
        # shutil.rmtree(old_ckpt)
        print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] delete ckpt {old_ckpt}.")
