#!/bin/bash

# download data https://tianchi.aliyun.com/dataset/131047
# save in {data_dir}, then split data.
split -l 500000 -d D1_0.csv ELE_Recommend_D1_0_split_

python build_model_sample.py --raw_file=${data_dir}/ELE_Recommend_D1_0_split_00 --res_file=${data_dir}/ELE_Recommend_D1_0_split_00_T
python build_model_sample.py --raw_file=${data_dir}/ELE_Recommend_D1_0_split_01 --res_file=${data_dir}/ELE_Recommend_D1_0_split_01_T
python build_model_sample.py --raw_file=${data_dir}/ELE_Recommend_D1_0_split_02 --res_file=${data_dir}/ELE_Recommend_D1_0_split_02_T
python build_model_sample.py --raw_file=${data_dir}/ELE_Recommend_D1_0_split_03 --res_file=${data_dir}/ELE_Recommend_D1_0_split_03_T
python build_model_sample.py --raw_file=${data_dir}/ELE_Recommend_D1_0_split_04 --res_file=${data_dir}/ELE_Recommend_D1_0_split_04_T
