if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

export CUDA_VISIBLE_DEVICES=0

ROOT_PATH="./dataset/weather/"
DATA_PATH="weather.csv"
MODEL="Modified_Transformer"
DATA="custom"
FEATURES="M"
SEQ_LEN=96
LABEL_LEN=48
E_LAYERS=2
D_LAYERS=1
FACTOR=3
ENC_IN=21
DEC_IN=21
C_OUT=21
DES="Exp"
ITR=1

BATCH_SIZE=256
DROP_OUT=0.05
lr=1e-5
EPOCHS=10
PATIENCE=3
# Transformer(baseline): 0, FANGated: 2, FAN: 4
EXP_SETTING=4 

for PRED_LEN in 96 192 336 720
do
    MODEL_ID="weather_${SEQ_LEN}_${PRED_LEN}"
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
        --batch_size $BATCH_SIZE \
        --dropout $DROP_OUT \
        --learning_rate $lr \
        --train_epochs $EPOCHS \
        --patience $PATIENCE \
        --exp_setting $EXP_SETTING # > logs/LongForecasting/$MODEL'_'$MODEL_ID'_'exp_setting_$EXP_SETTING.log 2>&1
done
