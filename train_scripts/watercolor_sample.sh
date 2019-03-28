#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python trainval_net_global_local.py --cuda --net res101 --dataset pascal_voc_water --dataset_t water --gc --lc --save_dir $2