#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

prefix=baseline_trans

python train.py \
    --batch_size 128 \
    --epochs 10 \
    --learning_rate 5e-5 \
    --prefix $prefix \
    --log_file logs/tmp.log
    # logs/$prefix.log # > logs/screen_$prefix.log 2>&1
    # --save_path checkpoints/$prefix # > logs/screen_$prefix.log 2>&1