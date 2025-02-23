# -*- coding: utf-8 -*-

import glob
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import csv
from typing import List, Dict, Iterator, Any

from tdrec.datasets.dataset import BaseDataset, BaseReader, FIELD_TYPE_TO_PA
from tdrec.protos.dataset_pb2 import DatasetConfig


class CsvDataset(BaseDataset):
    def __init__(self,
                 data_config: DatasetConfig,
                 input_path: str,
                 ):
        super().__init__(data_config, input_path)
        column_names = None
        column_types = {}

        for f in self._data_config.input_fields:
            if f.HasField("input_type"):
                column_types[f.input_name] = FIELD_TYPE_TO_PA[f.input_type]
            else:
                raise ValueError(
                    f"{f.input_type} of column [{f.input_name}] "
                    "is not supported by CsvDataset."
                )


class CsvReader(BaseReader):
    def __init__(self,
                 input_path: str,
                 batch_size: int,
                 column_names: List[str],
                 column_types: Dict[str, pa.DataType],
                 delimiter: str = ",",
                 ):
        super().__init__(input_path, batch_size)
        self._column_names = column_names
        self._csv_fmt = ds.CsvFileFormat(
            parse_options=csv.ParseOptions(delimiter=delimiter),
            convert_options=csv.ConvertOptions(column_types=column_types),
            read_options=csv.ReadOptions(
                column_names=column_names,
                # block_size=64 * 1024 * 1024,
            ),
        )

        self._input_files = []
        for input_path in self._input_path.split(","):
            self._input_files.extend(glob.glob(input_path))
        if len(self._input_files) == 0:
            raise RuntimeError(f"No csv files exist in {self._input_path}.")

    def to_batches(self) -> Iterator[Dict[str, pa.Array]]:
        input_files = self._input_files
        dataset = ds.dataset(input_files, format=self._csv_fmt)
        reader = dataset.to_batches(
            batch_size=self._batch_size, columns=self._column_names
        )
        yield from reader
