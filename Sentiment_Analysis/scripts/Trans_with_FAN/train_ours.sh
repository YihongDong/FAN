#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

num_hidden_layers=12
prefix=ours_trans

python train.py \
    --batch_size 128 \
    --epochs 50 \
    --learning_rate 5e-5 \
    --prefix $prefix \
    --replace_ffn \
    --num_hidden_layers $num_hidden_layers \
    --log_file logs/$prefix.log # > logs/screen_$prefix.log 2>&1
    # --save_path checkpoints/$prefix # > logs/screen_$prefix.log 2>&1