# -*- coding: utf-8 -*-

import math
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class BaseLR(LRScheduler):
    def __init__(self, optimizer: Optimizer):
        super().__init__(optimizer)

    def get_lr(self):
        return self.base_lrs



class ConstantLR(BaseLR):
    def __init__(self, optimizer: Optimizer):
        super().__init__(optimizer)


class ExponentialDecayLR(BaseLR):
    def __init__(self, optimizer: Optimizer,
                 decay_steps: int,
                 decay_rate: float,
                 min_learning_rate: float = 0.0,
                 ) -> None:
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._min_learning_rate = min_learning_rate
        super().__init__(optimizer)

    def get_lr(self):
        step_count = max(self._step_count - 1, 0)
        p = step_count / self._decay_steps
        scale = math.pow(self._decay_rate, p)
        lr = [
            max(base_lr * scale, self._min_learning_rate)
            for base_lr in self.base_lrs
        ]
        return lr
