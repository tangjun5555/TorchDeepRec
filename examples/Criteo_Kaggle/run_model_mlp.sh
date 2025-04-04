#!/bin/bash

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_split_00_T,${data_dir}/Criteo_Kaggle_split_01_T \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_01_T
# Mode.TRAIN Eval Result[model-3909]: auc:0.759465 copc:1.146153 ce_loss:0.525610
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_04_T
# Mode.EVALUATE Eval Result[model-3909]: auc:0.680582 copc:1.016026 ce_loss:0.465494

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_split_02_T,${data_dir}/Criteo_Kaggle_split_03_T \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_03_T
#
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_04_T
#

python -m tdrec.main \
  --task_type=export \
  --pipeline_config_path=model_mlp.config
