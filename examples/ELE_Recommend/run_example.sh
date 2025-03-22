#!/bin/bash

# download data https://tianchi.aliyun.com/dataset/131047
# save in {data_dir}, then split data.
split -l 500000 -d D1_0.csv ELE_Recommend_D1_0_split_
