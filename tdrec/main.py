# -*- coding: utf-8 -*-

import argparse
import os

from tdrec.constant import Mode
from tdrec.utils import config_util
from tdrec.features.feature import create_features
from tdrec.datasets.dataset import get_dataloader
from tdrec.models.base_model import create_model, TrainWrapper
from tdrec.pipeline import train_model, evaluate_model
from tdrec.utils import checkpoint_util
from tdrec.optimizer.optimizer_builder import create_optimizer


def train_and_evaluate(pipeline_config_path: str,
                       train_input_path: str,
                       eval_input_path: str,
                       continue_train: bool = True,
                       ):
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    model_dir = pipeline_config.model_dir

    dataset_config = pipeline_config.dataset_config
    features = create_features(list(pipeline_config.feature_configs))
    train_dataloader = get_dataloader(
        dataset_config=dataset_config,
        input_path=train_input_path,
        features=features,
        mode=Mode.TRAIN,
    )
    eval_dataloader = get_dataloader(
        dataset_config=dataset_config,
        input_path=eval_input_path,
        features=features,
        mode=Mode.EVALUATE,
    )

    model = create_model(
        pipeline_config.model_config,
        features,
        list(dataset_config.label_fields),
        sample_weight=dataset_config.sample_weight_field,
    )
    model = TrainWrapper(model)
    optimizer = create_optimizer(pipeline_config.train_config.optimizer_config)
    train_model(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        train_config=pipeline_config.train_config,
        model_dir=model_dir,
        ckpt_path=checkpoint_util.latest_checkpoint(model_dir)[0] if continue_train else None,
    )
    evaluate_model(
        model=model,
        eval_dataloader=eval_dataloader,
        eval_result_filename=os.path.join(model_dir, "train_eval_result.txt"),
    )


def evaluate(pipeline_config_path: str,
             eval_input_path: str,
             ):
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    model_dir = pipeline_config.model_dir

    dataset_config = pipeline_config.dataset_config
    features = create_features(list(pipeline_config.feature_configs))
    eval_dataloader = get_dataloader(
        dataset_config=dataset_config,
        input_path=eval_input_path,
        features=features,
        mode=Mode.EVALUATE,
    )

    model = create_model(
        pipeline_config.model_config,
        features,
        list(dataset_config.label_fields),
        sample_weight=dataset_config.sample_weight_field,
    )
    model = TrainWrapper(model)
    checkpoint_util.restore_model(
        checkpoint_dir=checkpoint_util.latest_checkpoint(model_dir)[0],
        model=model,
    )
    evaluate_model(
        model=model,
        eval_dataloader=eval_dataloader,
        eval_result_filename=os.path.join(pipeline_config.model_dir, "eval_result.txt"),
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
        "--eval_input_path",
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
        train_and_evaluate(
            pipeline_config_path=args.pipeline_config_path,
            train_input_path=args.train_input_path,
            eval_input_path=args.eval_input_path,
            continue_train=args.continue_train,
        )
    elif args.task_type == "evaluate":
        evaluate(
            pipeline_config_path=args.pipeline_config_path,
            eval_input_path=args.eval_input_path,
        )
    elif args.task_type == "export":
        raise NotImplementedError
    else:
        raise NotImplementedError
