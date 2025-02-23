# -*- coding: utf-8 -*-

import torch
import pyarrow as pa
from dataclasses import dataclass, field
from typing import Dict, Any, Iterator
from torch.utils.data import IterableDataset
from tdrec.protos.dataset_pb2 import DatasetConfig, FieldType


@dataclass
class Batch:
    features: Dict[str, torch.Tensor] = field(default_factory=dict)
    labels: Dict[str, torch.Tensor] = field(default_factory=dict)
    sample_weight: torch.Tensor = field(default_factory=torch.Tensor)


@dataclass
class ParsedData:
    name: str
    values: torch.Tensor


FIELD_TYPE_TO_PA = {
    FieldType.DOUBLE: pa.float64(),
    FieldType.INT64: pa.int64(),
    FieldType.STRING: pa.string(),
}


class BaseDataset(IterableDataset):
    def __init__(self,
                 data_config: DatasetConfig,
                 input_path: str,
                 ):
        super().__init__()
        self._data_config = data_config
        self._input_path = input_path



class BaseReader(object):
    def __init__(self,
                 input_path: str,
                 batch_size: int,
                 **kwargs: Any,
                 ):
        self._input_path = input_path
        self._batch_size = batch_size

    def to_batches(self) -> Iterator[Dict[str, pa.Array]]:
        """
        Get batch iterator.
        """
        raise NotImplementedError
