#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

export CUDA_VISIBLE_DEVICES=0

ROOT_PATH="./dataset/traffic/"
DATA_PATH="traffic.csv"
MODEL="Modified_Transformer"
DATA="custom"
FEATURES="M"
SEQ_LEN=96
LABEL_LEN=48
E_LAYERS=2
D_LAYERS=1
FACTOR=3
ENC_IN=862
DEC_IN=862
C_OUT=862
DES="Exp"
ITR=1
EXP_SETTING=2
TRAIN_EPOCHS=50
learn_rates=(1e-3 5e-5 1e-5 1e-6)

for LEARN_RATE in "${learn_rates[@]}"
do
    for PRED_LEN in 336 720
    do
        MODEL_ID="traffic_${SEQ_LEN}_${PRED_LEN}"
        
        python -u run.py \
            --is_training 1 \
            --root_path $ROOT_PATH \
            --data_path $DATA_PATH \
            --model_id $MODEL_ID \
            --model $MODEL \
            --data $DATA \
            --features $FEATURES \
            --seq_len $SEQ_LEN \
            --label_len $LABEL_LEN \
            --pred_len $PRED_LEN \
            --e_layers $E_LAYERS \
            --d_layers $D_LAYERS \
            --factor $FACTOR \
            --enc_in $ENC_IN \
            --dec_in $DEC_IN \
            --c_out $C_OUT \
            --des $DES \
            --itr $ITR \
            --train_epochs $TRAIN_EPOCHS \
            --exp_setting $EXP_SETTING \
            --learning_rate $LEARN_RATE >logs/LongForecasting/traffic/$MODEL'_'$MODEL_ID'_'expsetting$exp_setting'_'learn_rate_$LEARN_RATE.log 2>&1
    done
done
