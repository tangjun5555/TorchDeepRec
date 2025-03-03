# -*- coding: utf-8 -*-

import os
import pyarrow as pa
import unittest


class DatasetTest(unittest.TestCase):

    def test_csv_dataset(self):
        from tdrec.datasets.csv_dataset import CsvReader
        column_types = dict()
        column_types["y"] = pa.int64()
        for i in range(13):
            column_types["d" + str(i)] = pa.float64()
        for i in range(26):
            column_types["c" + str(i)] = pa.string()

        input_path = os.path.join(os.getenv("ROOT_DATA_DIR"), "Criteo_Kaggle/Criteo_Kaggle_split.00")
        print(f"数据目录:{input_path}")
        reader = CsvReader(
            input_path=input_path,
            batch_size=2,
            column_names=["y"] + ["d" + str(i) for i in range(13)] + ["c" + str(i) for i in range(26)],
            column_types=column_types,
            delimiter= "\t",
        )
        batchs = next(reader.to_batches())
        print(batchs)
