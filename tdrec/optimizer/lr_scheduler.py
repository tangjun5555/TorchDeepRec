# -*- coding: utf-8 -*-

import math

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tdrec.utils.load_class import get_register_class_meta

_LR_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_LR_CLASS_MAP)


class BaseLR(LRScheduler, metaclass=_meta_cls):
    def __init__(self, optimizer: Optimizer, by_epoch: bool = False):
        self._by_epoch = by_epoch
        super().__init__(optimizer)

    @property
    def by_epoch(self) -> bool:
        """Schedule by epoch or not."""
        return self._by_epoch


class ConstantLR(BaseLR):
    def __init__(self, optimizer: Optimizer):
        super().__init__(optimizer, by_epoch=True)
        print("Using ConstantLR.")

    def get_lr(self):
        """Calculates the learning rate."""
        return self.base_lrs


class ExponentialDecayLR(BaseLR):
    def __init__(self, optimizer: Optimizer,
                 decay_steps: int,
                 decay_rate: float,
                 min_learning_rate: float = 0.0,
                 by_epoch: bool = False
                 ) -> None:
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._min_learning_rate = min_learning_rate

        super().__init__(optimizer, by_epoch)
        print(f"Using ExponentialDecayLR, decay_steps:{decay_steps}, decay_rate:{decay_rate}.")

    def get_lr(self):
        step_count = max(self._step_count - 1, 0)
        p = step_count / self._decay_steps
        p = math.floor(p)
        scale = math.pow(self._decay_rate, p)
        lr = [
            max(base_lr * scale, self._min_learning_rate)
            for base_lr in self.base_lrs
        ]
        return lr
