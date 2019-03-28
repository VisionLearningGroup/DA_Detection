#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python test_net_global_local.py --cuda --net vgg16 --dataset cityscape_car --gc --lc --load_name $2