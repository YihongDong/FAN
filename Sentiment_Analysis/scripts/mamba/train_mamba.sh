#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model=mamba

hidden_layers=24
hidden_size=768
learning_rate=5e-5
prefix=$model'_'hlayers_$hidden_layers'_'hsize_$hidden_size'_'lr_$learning_rate'_'maxpooler

python train.py \
    --batch_size 128 \
    --epochs 50 \
    --learning_rate $learning_rate \
    --prefix $prefix \
    --num_hidden_layers $hidden_layers \
    --hidden_size $hidden_size \
    --max_pooler \
    --model $model \
    --log_file logs/$prefix.log # > logs/screen_$prefix.log 2>&1
    # --save_path checkpoints/$prefix # > logs/screen_$prefix.log 2>&1