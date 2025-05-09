#!/bin/bash

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp-dlrm.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_T_00 \
  --eval_input_path=${data_dir}/Criteo_Kaggle_T_00
# Mode.TRAIN Eval Result[model-7826]: auc:0.770082 copc:0.956744 ce_loss:0.459843
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp-dlrm.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_T_05
# Mode.EVALUATE Eval Result[model-7826]: auc:0.763224 copc:0.955872 ce_loss:0.487114

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp-dlrm.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_T_01 \
  --eval_input_path=${data_dir}/Criteo_Kaggle_T_01
#
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp-dlrm.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_T_05
#

python -m tdrec.main \
  --task_type=export \
  --pipeline_config_path=model_mlp-dlrm.config
