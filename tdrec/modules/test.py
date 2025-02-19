# -*- coding: utf-8 -*-

import torch


def test_mlp():
    from tdrec.modules.mlp import MLP
    mlp = MLP(
        in_features=16,
        hidden_units=[8, 4, 2],
        activation="torch.nn.ReLU",
        use_bn=True,
    )
    inputs = torch.randn(4, 16)
    result = mlp(inputs)
    print(f"inputs:{inputs}")
    print(f"result:{result}")

