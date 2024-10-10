#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model=mamba

hidden_layers=24
hidden_size=768
learning_rate=5e-5
prefix=$model'_'hlayers_$hidden_layers'_'hsize_$hidden_size'_'lr_$learning_rate'_'maxpooler

# dataset list
datasets=("imdb" "sentiment140" "amazon_polarity")

log_dir=./logs/$model
mkdir -p $log_dir

for dataset in "${datasets[@]}"
do
    python test.py \
        --batch_size 128 \
        --prefix $prefix \
        --dataset $dataset \
        --num_hidden_layers $hidden_layers \
        --hidden_size $hidden_size \
        --max_pooler \
        --model $model \
        --log_file $log_dir/$prefix'_'test_on_$dataset.log
done
