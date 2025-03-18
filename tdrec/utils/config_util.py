# -*- coding: utf-8 -*-

import os
from typing import Dict, Any
from google.protobuf import text_format, json_format
from google.protobuf.message import Message

from tdrec.protos.pipeline_pb2 import PipelineConfig


def load_pipeline_config(pipeline_config_path: str, allow_unknown_field: bool = False) -> PipelineConfig:
    config = PipelineConfig()
    with open(pipeline_config_path) as f:
        text_format.Merge(f.read(), config, allow_unknown_field=allow_unknown_field)
    return config


def config_to_kwargs(config: Message) -> Dict[str, Any]:
    """
    Convert a message to a config dict.
    """
    return json_format.MessageToDict(
        config,
        including_default_value_fields=True,
        preserving_proto_field_name=True,
    )

def save_message(message: Message, filepath: str) -> None:
    """
    Saves a proto message object to text file.
    Args:
        message: a proto message.
        filepath: save path.
    """
    directory, _ = os.path.split(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    pbtxt = text_format.MessageToString(message, as_utf8=True)
    with open(filepath, "w") as f:
        f.write(pbtxt)
