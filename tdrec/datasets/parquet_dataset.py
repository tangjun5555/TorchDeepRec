# -*- coding: utf-8 -*-

import glob
import random
import pyarrow as pa
import pyarrow.dataset as ds
from typing import List, Dict, Iterator, Any

from tdrec.datasets.dataset import BaseDataset, BaseReader
from tdrec.protos.dataset_pb2 import DatasetConfig


class ParquetDataset(BaseDataset):
    def __init__(self,
                 data_config: DatasetConfig,
                 input_path: str,
                 **kwargs: Any,
                 ):
        super().__init__(data_config, input_path, **kwargs)
        print(f"Using ParquetDataset.")

        column_names = []
        for f in self._dataset_config.input_fields:
            column_names.append(f.input_name)

        self._reader = ParquetReader(
            input_path=input_path,
            batch_size=self._batch_size,
            column_names=column_names,
        )


class ParquetReader(BaseReader):
    def __init__(self,
                 input_path: str,
                 batch_size: int,
                 column_names: List[str],
                 ):
        super().__init__(input_path, batch_size)
        self._column_names = column_names

        self._input_files = []
        for input_path in self._input_path.split(","):
            self._input_files.extend(glob.glob(input_path))
        if len(self._input_files) == 0:
            raise RuntimeError(f"No parquet files exist in {self._input_path}.")
        random.shuffle(self._input_files)
        print(f"input_files:{self._input_files}")

    def to_batches(self) -> Iterator[Dict[str, pa.Array]]:
        input_files = self._input_files
        dataset = ds.dataset(input_files, format="parquet")
        reader = dataset.to_batches(
            batch_size=self._batch_size,
            columns=self._column_names,
        )
        yield from reader
