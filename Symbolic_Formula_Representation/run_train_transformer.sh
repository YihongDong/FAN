dataset_ids=(0 1 2 3 4)

for dataset_id in ${dataset_ids[@]}
do
    CUDA_VISIBLE_DEVICES=0 python train_transformer.py \
        --dataset_id $dataset_id \
        --dataset_dir dataset \
        --save_dir transformer_checkpoint
        
done
