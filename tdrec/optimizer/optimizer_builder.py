# -*- coding: utf-8 -*-

import torch
from tdrec.protos.optimizer_pb2 import OptimizerConfig
from tdrec.utils.config_util import config_to_kwargs
from tdrec.optimizer.lr_scheduler import BaseLR


def create_optimizer(optimizer_config: OptimizerConfig):
    optimizer_type = optimizer_config.WhichOneof("optimizer")
    oneof_optimizer_config = getattr(optimizer_config, optimizer_type)
    optimizer_kwargs = config_to_kwargs(oneof_optimizer_config)

    if optimizer_type == "sgd_optimizer":
        return torch.optim.SGD, optimizer_kwargs
    elif optimizer_type == "adagrad_optimizer":
        return torch.optim.Adagrad, optimizer_kwargs
    elif optimizer_type == "adam_optimizer":
        beta1 = optimizer_kwargs.pop("beta1")
        beta2 = optimizer_kwargs.pop("beta2")
        optimizer_kwargs["betas"] = (beta1, beta2)
        return torch.optim.Adam, optimizer_kwargs
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer, optimizer_config: OptimizerConfig) -> BaseLR:
    lr_type = optimizer_config.WhichOneof("learning_rate")
    oneof_lr_config = getattr(optimizer_config, lr_type)
    lr_cls_name = oneof_lr_config.__class__.__name__
    lr_kwargs = config_to_kwargs(oneof_lr_config)
    lr_kwargs["optimizer"] = optimizer
    return BaseLR.create_class(lr_cls_name)(**lr_kwargs)
