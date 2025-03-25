#!/bin/bash

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp-fm.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_split_00_T \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_00_T
# Eval Result model-1955: auc:0.723142 copc:0.978027
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp-fm.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_04_T
# Eval Result model-1955: auc:0.658735 copc:0.925867
