#!/bin/bash

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_T_00,${data_dir}/Criteo_Kaggle_T_01 \
  --eval_input_path=${data_dir}/Criteo_Kaggle_T_01
# Mode.TRAIN Eval Result[model-31280]: auc:0.780717 copc:1.148334 ce_loss:0.545076
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_T_05
# Mode.EVALUATE Eval Result[model-31280]: auc:0.774522 copc:1.162139 ce_loss:0.487913

python -m tdrec.main \
  --task_type=export \
  --pipeline_config_path=model_mlp.config
