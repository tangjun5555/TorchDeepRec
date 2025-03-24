# -*- coding: utf-8 -*-

from typing import Dict
import numpy as np
import pyarrow as pa

import torch
from tdrec.constant import ParsedData
from tdrec.features.feature import BaseFeature
from tdrec.protos.feature_pb2 import FeatureUnit
from tdrec.utils.string_util import to_int_list


class TagFeature(BaseFeature):
    def __init__(self,
                 feature_config: FeatureUnit,
                 ):
        super().__init__(feature_config)
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.config.num_buckets,
            embedding_dim=self.config.embedding_dim,
            padding_idx=0,
        )

    def output_dim(self) -> int:
        return self.config.embedding_dim

    def parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        input_name = self.config.input_name
        values = input_data[input_name]
        if pa.types.is_string(values.type):
            res = []
            for row in values:
                res.append(to_int_list(str(row), self.config.separator))
            res = np.array(res)
            assert np.all(res > 0), f"feature[{self.name}] must be non negative."
        else:
            raise ValueError(
                f"feature[{self.name}] only support string input dtype now."
            )
        return ParsedData(name=self.name, values=torch.IntTensor(res))

    def to_dense(self, parsed_value: torch.Tensor) -> torch.Tensor:
        res = self.embedding(parsed_value)
        res = torch.sum(res, dim=1, keepdim=False)
        return res
