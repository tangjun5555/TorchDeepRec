#!/bin/bash

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp-din.config \
  --train_input_path=${data_dir}/D1_0_T \
  --eval_input_path=${data_dir}/D1_0_T
# Mode.TRAIN Eval Result[model-4250]: auc:0.573477 copc:0.551751 grouped_auc:0.569237 ce_loss:0.042137
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp-din.config \
  --eval_input_path=${data_dir}/D2_0_T
# Mode.EVALUATE Eval Result[model-4250]: auc:0.524431 copc:1.669765 grouped_auc:0.521632 ce_loss:0.120503

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp-din.config \
  --train_input_path=${data_dir}/D2_0_T \
  --eval_input_path=${data_dir}/D2_0_T
# Mode.TRAIN Eval Result[model-7885]: auc:0.577354 copc:1.116698 grouped_auc:0.568653 ce_loss:0.121089
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp-din.config \
  --eval_input_path=${data_dir}/D3_0_T
# Mode.EVALUATE Eval Result[model-7885]: auc:0.521457 copc:1.127030 grouped_auc:0.518227 ce_loss:0.142057

python -m tdrec.main \
  --task_type=train_and_evaluate \
  --pipeline_config_path=model_mlp-din.config \
  --train_input_path=${data_dir}/D3_0_T,${data_dir}/D4_0_T,${data_dir}/D5_0_T \
  --eval_input_path=${data_dir}/D5_0_T
# Mode.TRAIN Eval Result[model-21290]: auc:0.579794 copc:0.831560 grouped_auc:0.565372 ce_loss:0.014818
python -m tdrec.main \
  --task_type=evaluate \
  --pipeline_config_path=model_mlp-din.config \
  --eval_input_path=${data_dir}/D6_0_T
# Mode.EVALUATE Eval Result[model-21290]: auc:0.524913 copc:1.006227 grouped_auc:0.523283 ce_loss:0.109687

python -m tdrec.main \
  --task_type=export \
  --pipeline_config_path=model_mlp-din.config
