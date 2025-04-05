#!/bin/bash

# download data https://tianchi.aliyun.com/dataset/144733
# save in {data_dir}, then process data.

python build_model_sample.py --raw_file=${data_dir}/train.txt --res_file=${data_dir}/Criteo_Kaggle_T
split -l 8000000 -d ${data_dir}/Criteo_Kaggle_T ${data_dir}/Criteo_Kaggle_T_
