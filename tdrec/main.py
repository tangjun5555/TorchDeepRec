# -*- coding: utf-8 -*-

import argparse
from typing import List, Optional

from tdrec.constant import Mode
from tdrec.utils import config_util
from tdrec.protos.feature_pb2 import FeatureConfig
from tdrec.protos.dataset_pb2 import DatasetConfig
from tdrec.features.feature import create_features
from tdrec.datasets.dataset import get_dataloader


def train_and_evaluate(pipeline_config_path: str,
                       train_input_path: str,
                       eval_input_path: str,
                       continue_train: Optional[bool] = True,
                       ) -> None:
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    pipeline_config.train_input_path = train_input_path
    pipeline_config.eval_input_path = eval_input_path

    dataset_config = pipeline_config.dataset_config
    features = create_features(list(pipeline_config.feature_configs))
    train_dataloader = get_dataloader(
        dataset_config=dataset_config,
        input_path=pipeline_config.train_input_path,
        features=features,
        mode=Mode.TRAIN,
    )
    evaluate_dataloader = get_dataloader(
        dataset_config=dataset_config,
        input_path=pipeline_config.eval_input_path,
        features=features,
        mode=Mode.EVALUATE,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        help="Task type.",
        choices=["train_and_evaluate", "evaluate", "export"],
    )
    parser.add_argument(
        "--pipeline_config_path",
        type=str,
        default=None,
        help="Path to pipeline config file.",
    )
    parser.add_argument(
        "--train_input_path",
        type=str,
        default=None,
        help="train data input path",
    )
    parser.add_argument(
        "--evaluate_input_path",
        type=str,
        default=None,
        help="evaluate data input path",
    )
    parser.add_argument(
        "--continue_train",
        action="store_true",
        default=False,
        help="continue train using existing model_dir",
    )
    args, extra_args = parser.parse_known_args()

    if args.task_type == "train_and_evaluate":
        pass
    else:
        raise NotImplementedError
