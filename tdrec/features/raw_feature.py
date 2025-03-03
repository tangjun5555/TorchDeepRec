# -*- coding: utf-8 -*-

from typing import Dict
import pyarrow as pa

from tdrec.constant import ParsedData
from tdrec.features.feature import BaseFeature
from tdrec.protos.feature_pb2 import FeatureUnit


class RawFeature(BaseFeature):
    def __init__(self,
                 feature_config: FeatureUnit,
                 ):
        super().__init__(feature_config)

    def parse(self, input_data: Dict[str, pa.Array]) -> ParsedData:
        input_name = self.config.input_name
        values = input_data[input_name]
        if pa.types.is_floating(values.type):
            values = values.cast(pa.float32(), safe=False)
        else:
            raise ValueError(
                f"feature[{self.name}] only support double dtype now."
            )
        return ParsedData(name=self.name, values=values.to_numpy())
