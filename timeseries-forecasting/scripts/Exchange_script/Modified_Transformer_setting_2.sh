if [ ! -d "./logs/LongForecasting/exchange_rate" ]; then
    mkdir ./logs/LongForecasting/exchange_rate
fi

export CUDA_VISIBLE_DEVICES=0

ROOT_PATH="./dataset/exchange_rate/"
DATA_PATH="exchange_rate.csv"
MODEL="Modified_Transformer"
DATA="custom"
FEATURES=M
SEQ_LEN=96
LABEL_LEN=48
E_LAYERS=2
D_LAYERS=1
FACTOR=3
ENC_IN=8
DEC_IN=8
C_OUT=8
DES="Exp"
ITR=1
EXP_SETTING=2

# 96 192 336 720
for PRED_LEN in 96 192 336 720
do
    MODEL_ID="ECL_${SEQ_LEN}_${PRED_LEN}"
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
        --exp_setting $EXP_SETTING # >logs/LongForecasting/exchange_rate/$MODEL'_'$MODEL_ID'_'exp_setting_$EXP_SETTING.log 2>&1
done
