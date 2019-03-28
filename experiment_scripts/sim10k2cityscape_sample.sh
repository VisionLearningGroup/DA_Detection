#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python trainval_net_global_local.py --cuda --net vgg16 --dataset sim10k --dataset_t cityscape_car --gc --lc --save_dir $2