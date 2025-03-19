# -*- coding: utf-8 -*-

from typing import List, Optional, Dict
import numpy as np
import numpy.typing as npt
import pyarrow as pa

import torch
from tdrec.features.feature import BaseFeature
from tdrec.constant import Batch


def _to_tensor(x: npt.NDArray) -> torch.Tensor:
    if not x.flags.writeable:
        x = np.array(x)
    return torch.from_numpy(x)


class DataParser(object):
    def __init__(self,
                 features: List[BaseFeature],
                 labels: Optional[List[str]] = None,
                 sample_weight: Optional[str] = None,
                 is_training: bool = False,
                 ) -> None:
        self._features = features
        self._labels = labels or []
        self._sample_weight = sample_weight
        self._is_training = is_training

    def parse(self, input_data: Dict[str, pa.Array]) -> Dict[str, torch.Tensor]:
        output_data = {}

        for feature in self._features:
            feat_data = feature.parse(input_data)
            output_data[feature.name] = feat_data.values

        for label_name in self._labels:
            label = input_data[label_name]
            if pa.types.is_integer(label.type):
                output_data[label_name] = _to_tensor(
                    label.cast(pa.int64(), safe=False).to_numpy()
                )
            else:
                raise ValueError(
                    f"label column [{label_name}] only support int or float dtype now."
                )

        if self._sample_weight:
            values = input_data[self._sample_weight]
            output_data[self._sample_weight] = _to_tensor(
                values.cast(pa.float64(), safe=False).to_numpy()
            )
        return output_data

    def to_batch(self, input_data: Dict[str, torch.Tensor]) -> Batch:
        features = dict()
        for feature in self._features:
            features[feature.name] = input_data[feature.name]

        labels = dict()
        for label_name in self._labels:
            labels[label_name] = input_data[label_name]

        sample_weight: torch.Tensor = None
        if self._sample_weight:
            sample_weight = input_data[self._sample_weight]

        batch = Batch(
            features=features,
            labels=labels,
            sample_weight=sample_weight,
        )
        return batch
