#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

num_hidden_layers=12
prefix=ours_trans_withgate

datasets=("imdb")
# ( "imdb" "sentiment140" "amazon_polarity")

log_dir=./logs/$prefix
mkdir -p $log_dir

for dataset in "${datasets[@]}"
do
    python test.py \
        --batch_size 128 \
        --replace_ffn \
        --with_gate \
        --prefix $prefix \
        --dataset $dataset \
        --num_hidden_layers $num_hidden_layers \
        --log_file $log_dir/test_on_$dataset.log
done
