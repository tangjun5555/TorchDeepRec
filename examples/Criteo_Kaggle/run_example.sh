#!/bin/bash

# download data https://tianchi.aliyun.com/dataset/144733
# save in {data_dir}, then split data.
split -l 1000000 -d train.txt Criteo_Kaggle_split_

python build_model_sample.py --raw_file=${data_dir}/Criteo_Kaggle_split_45 --res_file=${data_dir}/Criteo_Kaggle_split_45_T

python build_model_sample.py --raw_file=${data_dir}/Criteo_Kaggle_split_00 --res_file=${data_dir}/Criteo_Kaggle_split_00_T
python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp_v1.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_split_00_T \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_00_T
# Eval Result model-2107: auc:0.763504 copc:1.034452
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp_v1.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_45_T
# Eval Result model-2107: auc:0.687388 copc:1.267633

python build_model_sample.py --raw_file=${data_dir}/Criteo_Kaggle_split_01 --res_file=${data_dir}/Criteo_Kaggle_split_01_T
python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp_v1.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_split_01_T \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_01_T
# Eval Result model-4214: auc:0.758623 copc:1.077758
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp_v1.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_45_T
#  Eval Result model-4214: auc:0.679035 copc:1.095662

python build_model_sample.py --raw_file=${data_dir}/Criteo_Kaggle_split_02 --res_file=${data_dir}/Criteo_Kaggle_split_02_T
python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp_v1.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_split_02_T \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_02_T
# Eval Result model-6331: auc:0.758502 copc:0.951271
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp_v1.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_45_T
# Eval Result model-6331: auc:0.681520 copc:0.904874

python build_model_sample.py --raw_file=${data_dir}/Criteo_Kaggle_split_03 --res_file=${data_dir}/Criteo_Kaggle_split_03_T
python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp_v1.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_split_03_T \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_03_T
# Eval Result model-8442: auc:0.758895 copc:0.981576
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp_v1.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_split_45_T
# Eval Result model-8442: auc:0.692347 copc:0.790076

python -m tdrec.main \
  --task_type=export \
  --pipeline_config_path=model_mlp_v1.config
