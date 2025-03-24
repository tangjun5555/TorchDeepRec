#!/bin/bash

# download data https://tianchi.aliyun.com/dataset/144733
# save in {data_dir}, then split data.
split -l 1000000 -d train.txt Criteo_Kaggle_split_

python build_model_sample.py --raw_file=${data_dir}/Criteo_Kaggle_split_00 --res_file=${data_dir}/Criteo_Kaggle_split_00_T
python build_model_sample.py --raw_file=${data_dir}/Criteo_Kaggle_split_01 --res_file=${data_dir}/Criteo_Kaggle_split_01_T
python build_model_sample.py --raw_file=${data_dir}/Criteo_Kaggle_split_02 --res_file=${data_dir}/Criteo_Kaggle_split_02_T
python build_model_sample.py --raw_file=${data_dir}/Criteo_Kaggle_split_03 --res_file=${data_dir}/Criteo_Kaggle_split_03_T
python build_model_sample.py --raw_file=${data_dir}/Criteo_Kaggle_split_04 --res_file=${data_dir}/Criteo_Kaggle_split_04_T
