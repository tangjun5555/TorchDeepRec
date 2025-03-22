# -*- coding: utf-8 -*-

import pyarrow as pa
from abc import abstractmethod
from typing import Dict, Any, Iterator, List

import torch
from torch.utils.data import IterableDataset, DataLoader
from tdrec.protos.dataset_pb2 import DatasetConfig, FieldType, DatasetType
from tdrec.constant import Mode
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

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for input_data in self._reader.to_batches():
            yield self._build_batch(input_data)

    def _build_batch(self, input_data: Dict[str, pa.Array]) -> Dict[str, torch.Tensor]:
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

def get_dummy_inputs(dataset_config: DatasetConfig):
    exclude_fields = list(dataset_config.label_fields)
    if dataset_config.sample_weight_field:
        exclude_fields.append(dataset_config.sample_weight_field)
    dummy_inputs = dict()
    input_names = []
    batch_size = 2
    for input_field in dataset_config.input_fields:
        if input_field.input_name in exclude_fields:
            continue
        input_names.append(input_field.input_name)
        if input_field.input_type == FieldType.DOUBLE:
            dummy_inputs[input_field.input_name] = torch.randn((batch_size, 1))
        elif input_field.input_type == FieldType.INT:
            dummy_inputs[input_field.input_name] = torch.randint(low=1, high=100, size=(batch_size,))
        elif input_field.input_type == FieldType.String:
            assert input_field.split_length > 0
            assert input_field.sub_type in ["DOUBLE", "INT"]
            if input_field.sub_type == "DOUBLE":
                dummy_inputs[input_field.input_name] = torch.randn((batch_size, input_field.split_length))
            elif input_field.sub_type == "INT":
                dummy_inputs[input_field.input_name] = torch.randint(low=1, high=100, size=(batch_size, input_field.split_length))
            else:
                raise ValueError(f"sub_type:{input_field.sub_type} is not supported.")
        else:
            raise ValueError(f"input_type:{input_field.input_type} is not supported.")
    for input_field in dataset_config.label_fields:
        dummy_inputs[input_field] = torch.randint(low=1, high=100, size=(batch_size,))
    if dataset_config.sample_weight_field:
        dummy_inputs[dataset_config.sample_weight_field] = torch.randn((batch_size,))
    return {"inputs": dummy_inputs}, input_names
