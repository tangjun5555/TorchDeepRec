# -*- coding: utf-8 -*-

from typing import Dict
import pyarrow as pa

import torch
from tdrec.constant import ParsedData
from tdrec.features.feature import BaseFeature
from tdrec.protos.feature_pb2 import FeatureUnit


class IdFeature(BaseFeature):
    def __init__(self,
                 feature_config: FeatureUnit,
                 ):
        super().__init__(feature_config)

    def parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        input_name = self.config.input_name
        values = input_data[input_name]
        if pa.types.is_integer(values.type):
            values = values.cast(pa.int64(), safe=False)
        else:
            raise ValueError(
                f"feature[{self.name}] only support int dtype input now."
            )
        return ParsedData(name=self.name, values=torch.Tensor(values.to_numpy()))

    def to_dense(self, parsed_value: torch.Tensor) -> torch.Tensor:
        embedding = torch.nn.Embedding(
            num_embeddings=self.config.num_buckets,
            embedding_dim=self.config.embedding_dim,
        )
        return embedding(parsed_value)
