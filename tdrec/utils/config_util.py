# -*- coding: utf-8 -*-

from tdrec.protos.pipeline_pb2 import PipelineConfig
from google.protobuf import text_format


def load_pipeline_config(pipeline_config_path: str, allow_unknown_field: bool = False) -> PipelineConfig:
    config = PipelineConfig()
    with open(pipeline_config_path) as f:
        text_format.Merge(f.read(), config, allow_unknown_field=allow_unknown_field)
    return config
