# -*- coding: utf-8 -*-

import argparse
import os
import time
import datetime

import torch

from tdrec.constant import Mode
from tdrec.version import __version__ as tdrec_version
from tdrec.utils import config_util
from tdrec.features.feature import create_features
from tdrec.datasets.dataset import get_dataloader, get_dummy_inputs
from tdrec.models.base_model import create_model, TrainWrapper
from tdrec.pipeline import train_model, evaluate_model
from tdrec.utils import checkpoint_util
from tdrec.optimizer.optimizer_builder import create_optimizer, create_scheduler


def train_and_evaluate(pipeline_config_path: str,
                       train_input_path: str,
                       eval_input_path: str,
                       ):
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    model_dir = pipeline_config.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"create dir {model_dir}")

    dataset_config = pipeline_config.dataset_config
    features = create_features(list(pipeline_config.feature_config.features))
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
    optimizer_cls, optimizer_kwargs = create_optimizer(pipeline_config.train_config.optimizer_config)
    optimizer = optimizer_cls(params=model.model.parameters(), **optimizer_kwargs)
    lr_scheduler = create_scheduler(optimizer, pipeline_config.train_config.optimizer_config)
    checkpoint_util.restore_model(
        checkpoint_dir=checkpoint_util.latest_checkpoint(model_dir)[0],
        model=model,
        optimizer=optimizer,
    )
    global_step = checkpoint_util.latest_checkpoint(model_dir)[1]
    print(f"global_step:{global_step}")

    global_step = train_model(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        train_config=pipeline_config.train_config,
        model_dir=model_dir,
        global_step=global_step,
    )
    evaluate_model(
        model=model,
        eval_dataloader=eval_dataloader,
        model_dir=model_dir,
        eval_result_filename=os.path.join(model_dir, "train_eval_result.txt"),
        global_step=global_step,
    )

    config_util.save_message(
        pipeline_config, os.path.join(pipeline_config.model_dir, "pipeline.config")
    )
    with open(os.path.join(pipeline_config.model_dir, "version"), "w") as f:
        f.write(tdrec_version + "\n")


def evaluate(pipeline_config_path: str,
             eval_input_path: str,
             ):
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    model_dir = pipeline_config.model_dir

    dataset_config = pipeline_config.dataset_config
    features = create_features(list(pipeline_config.feature_config.features))
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
    global_step = checkpoint_util.latest_checkpoint(model_dir)[1]
    print(f"global_step:{global_step}")

    evaluate_model(
        model=model,
        eval_dataloader=eval_dataloader,
        model_dir=model_dir,
        eval_result_filename=os.path.join(pipeline_config.model_dir, "eval_result.txt"),
        global_step=global_step,
    )


def export(pipeline_config_path: str):
    pipeline_config = config_util.load_pipeline_config(pipeline_config_path)
    model_dir = pipeline_config.model_dir

    export_dir = os.path.join(model_dir, "export")
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print(f"create dir {export_dir}")

    dataset_config = pipeline_config.dataset_config
    features = create_features(list(pipeline_config.feature_config.features))

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

    dummy_inputs, input_names = get_dummy_inputs(dataset_config)
    dynamic_axes = dict()
    for input_name in input_names:
        dynamic_axes[input_name] = {0 : "batch_size"}
    dynamic_axes["outputs"] = {0 : "batch_size"}
    export_file = os.path.join(export_dir, f"model-{int(time.time())}.onnx")
    torch.onnx.export(
        model,
        dummy_inputs,
        export_file,
        input_names=input_names,
        output_names=["outputs"],
        dynamic_axes=dynamic_axes,
    )
    print(f"[INFO] [{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Successfully export model to {export_file}.")


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
        required=True,
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
    args, extra_args = parser.parse_known_args()

    if args.task_type == "train_and_evaluate":
        train_and_evaluate(
            pipeline_config_path=args.pipeline_config_path,
            train_input_path=args.train_input_path,
            eval_input_path=args.eval_input_path,
        )
    elif args.task_type == "evaluate":
        evaluate(
            pipeline_config_path=args.pipeline_config_path,
            eval_input_path=args.eval_input_path,
        )
    elif args.task_type == "export":
        export(pipeline_config_path=args.pipeline_config_path)
    else:
        raise NotImplementedError
