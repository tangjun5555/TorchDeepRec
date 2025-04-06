#!/bin/bash

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_T_00 \
  --eval_input_path=${data_dir}/Criteo_Kaggle_T_00
# Mode.TRAIN Eval Result[model-7826]: auc:0.777409 copc:0.960497 ce_loss:0.457538
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_T_05
# Mode.EVALUATE Eval Result[model-7826]: auc:0.770061 copc:0.960418 ce_loss:0.484276

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp.config \
  --train_input_path=${data_dir}/Criteo_Kaggle_T_01 \
  --eval_input_path=${data_dir}/Criteo_Kaggle_T_01
# Mode.TRAIN Eval Result[model-15654]: auc:0.777881 copc:1.181039 ce_loss:0.536273
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp.config \
  --eval_input_path=${data_dir}/Criteo_Kaggle_T_05
# Mode.EVALUATE Eval Result[model-15654]: auc:0.771365 copc:1.196604 ce_loss:0.498744

python -m tdrec.main \
  --task_type=export \
  --pipeline_config_path=model_mlp.config
