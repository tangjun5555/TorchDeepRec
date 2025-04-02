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

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp-dlrm.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_split_01_T \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_01_T
# Mode.TRAIN Eval Result[model-3910]: auc:0.753545 copc:1.042839
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp-dlrm.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_04_T
# Mode.EVALUATE Eval Result[model-3910]: auc:0.669516 copc:1.191942

python -m tdrec.main \
  --task_type=export \
  --pipeline_config_path=model_mlp-dlrm.config
