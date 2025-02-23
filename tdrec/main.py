# -*- coding: utf-8 -*-

import argparse
from typing import List, Optional
from tdrec.utils import config_util
from tdrec.protos.feature_pb2 import FeatureConfig
from tdrec.protos.dataset_pb2 import DatasetConfig
from tdrec.features.feature import create_features


def train_and_evaluate(pipeline_config_path: str,
                       train_input_path: str,
                       eval_input_path: str,
                       model_dir: Optional[str] = None,
                       continue_train: Optional[bool] = True,
    ) -> None:
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    pipeline_config.train_input_path = train_input_path
    pipeline_config.eval_input_path = eval_input_path
    if model_dir:
        pipeline_config.model_dir = model_dir

    dataset_config = pipeline_config.dataset_config
    features = create_features(list(pipeline_config.feature_configs))



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
        "--eval_input_path",
        type=str,
        default=None,
        help="eval data input path",
    )
    parser.add_argument(
        "--continue_train",
        action="store_true",
        default=False,
        help="continue train using existing model_dir",
    )
    args, extra_args = parser.parse_known_args()

    if args.task_type == "train_and_evaluate":
        train_and_evaluate(

        )
    else:
        raise NotImplementedError
