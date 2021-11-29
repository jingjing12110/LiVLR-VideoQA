#! /bin/bash

CUDA_VISIBLE_DEVICES=1,2 python train.py --bs 256 --n_works 8 --lr 0.00008 --exp_name Exp_DiVS/all --num_head 8 --epoch 80
#CUDA_VISIBLE_DEVICES=0 python train_msrvtt.py --bs 128 --n_works 8 --lr 0.00008 --exp_name Exp_DiVS/all --num_head 8 --epoch 100
