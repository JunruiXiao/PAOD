#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-29500}

# paod_r50
./tools/dist_train.sh configs/paod/paod_r50_fpn_1x_voc.py 4

# paod_r101
./tools/dist_train.sh configs/paod/paod_r101_fpn_1x_voc.py 4