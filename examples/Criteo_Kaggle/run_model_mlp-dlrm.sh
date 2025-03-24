#!/bin/bash

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp-dlrm.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_split_00_T \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_00_T
# Eval Result model-1955: auc:0.749777 copc:0.946836
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp-dlrm.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_04_T
# Eval Result model-1955: auc:0.682062 copc:1.245979
