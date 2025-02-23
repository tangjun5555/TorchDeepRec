# -*- coding: utf-8 -*-

import pyarrow as pa
import numpy.typing as npt
from dataclasses import dataclass, field
from abc import abstractmethod
from typing import Dict, Any, Iterator, List

import torch
from torch.utils.data import IterableDataset
from tdrec.protos.dataset_pb2 import DatasetConfig, FieldType
from tdrec.constant import Mode
from tdrec.datasets.data_parser import DataParser
from tdrec.features.feature import BaseFeature


@dataclass
class Batch:
    features: Dict[str, torch.Tensor] = field(default_factory=dict)
    labels: Dict[str, torch.Tensor] = field(default_factory=dict)
    sample_weight: torch.Tensor = field(default_factory=torch.Tensor)


@dataclass
class ParsedData:
    name: str
    values: npt.NDArray


FIELD_TYPE_TO_PA = {
    FieldType.DOUBLE: pa.float32(),
    FieldType.INT: pa.int32(),
    FieldType.STRING: pa.string(),
}


class BaseDataset(IterableDataset):
    def __init__(self,
                 dataset_config: DatasetConfig,
                 input_path: str,
                 features: List[BaseFeature],
                 mode: Mode = Mode.EVALUATE,
                 ):
        super().__init__()
        self._dataset_config = dataset_config
        self._input_path = input_path
        self._features = features
        self._mode = mode

        self._batch_size = dataset_config.batch_size
        self._reader = None

        self._data_parser = DataParser(
            features=features,
            labels=list(dataset_config.label_fields),
            sample_weight=dataset_config.sample_weight_field,
            is_training=mode == Mode.TRAIN,
        )

    def __iter__(self) -> Iterator[Batch]:
        for input_data in self._reader.to_batches():
            yield self._build_batch(input_data)

    def _build_batch(self, input_data: Dict[str, pa.Array]) -> Batch:
        output_data = self._data_parser.parse(input_data)

        if self._mode == Mode.PREDICT:
            batch = self._data_parser.to_batch(output_data)
            # TODO
        else:
            batch = self._data_parser.to_batch(output_data)
        return batch


class BaseReader(object):
    def __init__(self,
                 input_path: str,
                 batch_size: int,
                 **kwargs: Any,
                 ):
        self._input_path = input_path
        self._batch_size = batch_size

    @abstractmethod
    def to_batches(self) -> Iterator[Dict[str, pa.Array]]:
        """
        Get batch iterator.
        """
        raise NotImplementedError
