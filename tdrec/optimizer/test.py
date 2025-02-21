# -*- coding: utf-8 -*-

import torch
import unittest


class LRSchedulerTest(unittest.TestCase):

    def test_constant_lr(self) -> None:
        from tdrec.optimizer.lr_scheduler import ConstantLR
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.01)
        lr = ConstantLR(opt)
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.01)

    def test_exponential_decay_lr(self) -> None:
        from tdrec.optimizer.lr_scheduler import ExponentialDecayLR
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.01)
        lr = ExponentialDecayLR(
            opt, decay_steps=1, decay_rate=0.7
        )
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.007)
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.0049)
