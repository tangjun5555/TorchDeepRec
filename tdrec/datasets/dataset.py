# -*- coding: utf-8 -*-

import pyarrow as pa
from abc import abstractmethod
from typing import Dict, Any, Iterator, List

from torch.utils.data import IterableDataset, DataLoader
from tdrec.protos.dataset_pb2 import DatasetConfig, FieldType, DatasetType
from tdrec.constant import Mode, Batch
from tdrec.datasets.data_parser import DataParser
from tdrec.features.feature import BaseFeature
from tdrec.utils.load_class import get_register_class_meta


_DATASET_CLASS_MAP = {}
_dataset_meta_cls = get_register_class_meta(_DATASET_CLASS_MAP)
FIELD_TYPE_TO_PA = {
    FieldType.DOUBLE: pa.float64(),
    FieldType.INT: pa.int64(),
    FieldType.STRING: pa.string(),
}


class BaseDataset(IterableDataset, metaclass=_dataset_meta_cls):
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


def get_dataloader(dataset_config: DatasetConfig,
                   input_path: str,
                   features: List[BaseFeature],
                   mode: Mode = Mode.EVALUATE,
                   ) -> DataLoader:
    dataset_name = DatasetType.Name(dataset_config.dataset_type)
    dataset_cls = BaseDataset.create_class(dataset_name)
    dataset = dataset_cls(
        data_config=dataset_config,
        input_path=input_path,
        features=features,
        mode=mode,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
    )
    return dataloader

def build_dummy_input(dataset_config: DatasetConfig):
    pass
