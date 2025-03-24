# -*- coding: utf-8 -*-

from typing import Dict
import pyarrow as pa
import numpy as np

import torch
from tdrec.constant import ParsedData
from tdrec.features.feature import BaseFeature
from tdrec.protos.feature_pb2 import FeatureUnit


class IdFeature(BaseFeature):
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
        if pa.types.is_integer(values.type):
            values = values.cast(pa.int64(), safe=False)
            values = np.array(values.to_numpy())
            assert np.all(values > 0), f"feature[{self.name}] must be non negative."
        else:
            raise ValueError(
                f"feature[{self.name}] only support int dtype input now."
            )
        return ParsedData(name=self.name, values=torch.IntTensor(values))

    def to_dense(self, parsed_value: torch.Tensor) -> torch.Tensor:
        return self.embedding(parsed_value)
