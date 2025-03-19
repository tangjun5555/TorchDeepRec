# -*- coding: utf-8 -*-

from typing import Dict
import numpy as np
import pyarrow as pa

import torch
from tdrec.constant import ParsedData
from tdrec.features.feature import BaseFeature
from tdrec.protos.feature_pb2 import FeatureUnit
from tdrec.utils.string_util import to_float_list


class RawFeature(BaseFeature):
    def __init__(self,
                 feature_config: FeatureUnit,
                 ):
        super().__init__(feature_config)

    def parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        input_name = self.config.input_name
        values = input_data[input_name]
        if pa.types.is_floating(values.type):
            values = values.cast(pa.float64(), safe=False)
            res = np.array(values)
            res = np.reshape(res, (-1, 1))
        elif pa.types.is_integer(values.type):
            res = []
            for row in values:
                res.append(to_float_list(str(row), self.config.separator))
            res = np.array(res)
        else:
            raise ValueError(
                f"feature[{self.name}] only support double|string dtype input now."
            )
        return ParsedData(name=self.name, values=torch.FloatTensor(res))

    def to_dense(self, parsed_value: torch.Tensor) -> torch.Tensor:
        if self.config.embedding_dim:
            return torch.nn.Linear(self.config.value_dim, self.config.embedding_dim)(parsed_value)
        else:
            return parsed_value
