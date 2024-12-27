# !/bin/bash


export FORCE_TORCHRUN=1
export CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6,7


cd ../LLaMA-Factory
pip install .
pip install deepspeed==0.14.4 accelerate==1.0.1 transformers==4.42.1


datasets=(
    budget_adjusted_v2_1223_cleaned_5sets_euclidean_nonlinear_v4_0.95_alpha_0.3_lambda_0.9_gamma_2.0_6k_self_instruct
    budget_adjusted_v2_1223_cleaned_5sets_euclidean_nonlinear_v4_0.95_alpha_0.3_lambda_0.9_gamma_2.0_6k_alpaca
    budget_adjusted_v2_1223_cleaned_5sets_euclidean_nonlinear_v4_0.95_alpha_0.3_lambda_0.9_gamma_2.0_6k_dolly
    budget_adjusted_v2_1223_cleaned_5sets_euclidean_nonlinear_v4_0.95_alpha_0.3_lambda_0.9_gamma_2.0_6k_sharegpt
)

cd ../LLaMA-Factory
MODEL=../../allmodels/Meta-Llama-3-8B-base-original
for data in "${datasets[@]}"
do 

    OUT_DIR="../saves/checkpoints/llama3_8b_${data}"
    echo "$OUT_DIR"
    deepspeed --master_port 29502 --num_gpus=8 src/train.py \
    --model_name_or_path "${MODEL}"  \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed ../LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --dataset "${data}" \
    --template llama3 \
    --cutoff_len 2048 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir "${OUT_DIR}" \
    --logging_steps 10 \
    --save_strategy no \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 6.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --fp16 \
    --seed 8 \
    --data_seed 8 \
    --ddp_timeout 180000000
done



cd ../LLaMA-Factory
MODEL=../../allmodels/Qwen2.5-7B
for data in "${datasets[@]}"
do
    OUT_DIR="../saves/checkpoints/qwen7b_${data}"
    echo "$OUT_DIR"
    deepspeed --num_gpus=8 src/train.py \
    --model_name_or_path "${MODEL}"  \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed ../LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --dataset "${data}" \
    --template qwen \
    --cutoff_len 2048 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir "${OUT_DIR}" \
    --logging_steps 10 \
    --save_strategy no \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 6.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --fp16 \
    --seed 8 \
    --data_seed 8 \
    --ddp_timeout 180000000
done

cd ../LLaMA-Factory
MODEL=../../allmodels/mistral-7b-v0.3
for data in "${datasets[@]}"
do
    OUT_DIR="../saves/checkpoints/robust_mistral7b_${data}"
    echo "$OUT_DIR"
    deepspeed --master_port 29502 --num_gpus=8 src/train.py \
    --model_name_or_path "${MODEL}"  \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --deepspeed ../LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --dataset "${data}" \
    --template mistral \
    --cutoff_len 2048 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir "${OUT_DIR}" \
    --logging_steps 10 \
    --save_strategy epoch \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 3.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --fp16 \
    --seed 8 \
    --data_seed 8 \
    --ddp_timeout 180000000
done
