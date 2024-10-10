#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

prefix=ours_trans_withgate

num_hidden_layers=12

python train.py \
    --batch_size 128 \
    --epochs 50 \
    --learning_rate 5e-5 \
    --prefix $prefix \
    --replace_ffn \
    --with_gate \
    --num_hidden_layers $num_hidden_layers \
    --log_file logs/$prefix'_'hlayers_$num_hidden_layers.log # > logs/screen_$prefix.log 2>&1
    # --save_path checkpoints/$prefix # > logs/screen_$prefix.log 2>&1