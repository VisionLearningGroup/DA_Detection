#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python test_net_global.py --cuda --net res101 --dataset clipart --gc --load_name $2