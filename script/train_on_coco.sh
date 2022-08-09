#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-29500}

# # r50_1x_coco
# ./tools/dist_train.sh configs/paod/paod_r50_fpn_1x_coco.py 4

# # r50_Mstrian_2x_coco
# ./tools/dist_train.sh configs/paod/paod_r50_fpn_mstrain_2x_coco.py 4

# # r101_Mstrian_2x_coco
# ./tools/dist_train.sh configs/paod/paod_r101_fpn_mstrain_2x_coco.py 4

# # r101_dcn_Mstrian_2x_coco
# ./tools/dist_train.sh configs/paod/paod_r101_fpn_dcn_mstrain_2x_coco.py 4

# # X-101-64x4d_Mstrian_2x_coco
# ./tools/dist_train.sh configs/paod/paod_x101_64x4d_fpn_mstrain_2x_coco.py 4

# # X-101-64x4d_dcn_Mstrian_2x_coco
# ./tools/dist_train.sh configs/paod/paod_x101_64x4d_fpn_dcn_mstrain_2x_coco.py 4

# # r2-101_Mstrian_2x_coco
# ./tools/dist_train.sh configs/paod/paod_r2101_fpn_mstrain_2x_coco.py 4

# r2-101_dcn_Mstrian_2x_coco
./tools/dist_train.sh configs/paod/paod_r2101_fpn_dcn_mstrain_2x_coco.py 4