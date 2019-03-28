#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python trainval_net_global.py --cuda --net res101 --dataset pascal_voc_0712 --dataset_t clipart --gc --save_dir $2