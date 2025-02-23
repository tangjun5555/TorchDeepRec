# -*- coding: utf-8 -*-

from typing import Dict
import pyarrow as pa

from tdrec.datasets.dataset import ParsedData
from tdrec.features.feature import BaseFeature
from tdrec.protos.feature_pb2 import FeatureConfig


class IdFeature(BaseFeature):
    def __init__(self,
                 feature_config: FeatureConfig,
                 ):
        super().__init__(feature_config)

    def parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        input_name = self.config.input_names[0]
        values = input_data[input_name]
        if pa.types.is_integer(values.type):
            values = values.cast(pa.int32(), safe=False)
        elif pa.types.is_string(values.type):
            values = values.cast(pa.string(), safe=False)
        else:
            raise ValueError(
                f"feature column [{input_name}] only support int or string dtype now."
            )
        return ParsedData(name=self.name, values=values.to_numpy())
