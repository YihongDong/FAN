#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

num_hidden_layers=12
prefix=ours_trans

dataset=amazon_polarity # (imdb, sentiment140, amazon_polarity)

# create log dir
log_dir=./logs/$prefix
mkdir -p $log_dir

python test.py \
    --batch_size 128 \
    --replace_ffn \
    --prefix $prefix \
    --dataset $dataset \
    --num_hidden_layers $num_hidden_layers \
    --log_file $log_dir/test_on_$dataset.log # > logs/screen_$prefix.log 2>&1
    # --save_path checkpoints/$prefix # > logs/screen_$prefix.log 2>&1