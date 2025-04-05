#!/bin/bash

# download data https://tianchi.aliyun.com/dataset/131047
# save in {data_dir}, then process data.

python build_model_sample.py --raw_file=${data_dir}/D1_0.csv --res_file=${data_dir}/D1_0_T
python build_model_sample.py --raw_file=${data_dir}/D2_0.csv --res_file=${data_dir}/D2_0_T
python build_model_sample.py --raw_file=${data_dir}/D3_0.csv --res_file=${data_dir}/D3_0_T
python build_model_sample.py --raw_file=${data_dir}/D4_0.csv --res_file=${data_dir}/D4_0_T
python build_model_sample.py --raw_file=${data_dir}/D5_0.csv --res_file=${data_dir}/D5_0_T
python build_model_sample.py --raw_file=${data_dir}/D6_0.csv --res_file=${data_dir}/D6_0_T
