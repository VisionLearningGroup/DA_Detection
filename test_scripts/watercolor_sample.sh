#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python test_net_global_local.py --cuda --net res101 --dataset water --gc --lc --load_name $2