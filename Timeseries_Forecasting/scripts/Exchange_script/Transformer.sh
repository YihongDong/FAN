export CUDA_VISIBLE_DEVICES=0

model_id_prefix="Exchange_96"

for pred_len in 96 192 336 720; do
  model_id="${model_id_prefix}_${pred_len}"
  
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id "$model_id" \
    --model Modified_Transformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "$pred_len" \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --exp_setting 0 \
    --itr 1 >logs/LongForecasting/exchange_rate/baseline_$model_id.log 2>&1
done