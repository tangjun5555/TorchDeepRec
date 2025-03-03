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
            values = values.cast(pa.int32(), safe=False)
        elif pa.types.is_string(values.type):
            values = values.cast(pa.string(), safe=False)
        else:
            raise ValueError(
                f"feature[{self.name}] only support int or string dtype now."
            )
        return ParsedData(name=self.name, values= torch.Tensor(values.to_numpy()))

    def parse_from_raw_data(self, input_data: Dict[str, torch.Tensor]) -> ParsedData:
        input_name = self.config.input_name
        values = input_data[input_name]

        # if values.dtype == torch.


