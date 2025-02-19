# -*- coding: utf-8 -*-

def test_load_by_path():
    from tdrec.utils.load_class import load_by_path
    loaded_cls = load_by_path("torch.nn.ReLU")
    print(loaded_cls)
