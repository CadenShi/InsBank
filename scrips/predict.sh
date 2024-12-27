#!/bin/bash
cd ../LLaMA-Factory

export CUDA_VISIBLE_DEVICES=0

cd ../LLaMA-Factory
pip install .
pip install deepspeed==0.14.4 accelerate==1.0.1 transformers==4.42.1

cd ../FastChat
pip install .
pip install anthropic==0.31.2


datasets=(
    robust_mistral7b_cleaned_full
)

for data in "${datasets[@]}"
do
    MODEL_PATH="../saves/checkpoints/${data}"
    MODEL_ID=mistral
    ANSWER_FILE="../saves/mtbench/${data}.jsonl"
    cd ../FastChat/fastchat/llm_judge
    python gen_model_answer.py --model-path $MODEL_PATH --model-id $MODEL_ID --answer-file $ANSWER_FILE --max-new-token 512

    MODEL_DIR="../saves/checkpoints/${data}"
    SAVE_DIR="../saves/alpacaeval/${data}"
    cd ../LLaMA-Factory
    torchrun src/train.py \
    --model_name_or_path "${MODEL_DIR}" \
    --stage sft \
    --do_predict \
    --finetuning_type full \
    --eval_dataset alpaca_eval \
    --template mistral \
    --cutoff_len 2048 \
    --max_samples 1000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir "${SAVE_DIR}" \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
    --fp16 \
    --ddp_timeout 180000000 \
    --temperature 0.7 \
    --top_p 0.9 \
    --num_beams 1 \
    --max_length 2048 \
    --max_new_tokens 512 \
    --seed 8
done


datasets=(
    llama3_8b_budget_adjusted_v2_first_2k_1223_cleaned_5sets_euclidean_nonlinear_v4_0.95_alpha_0.3_lambda_0.9_gamma_2.0_6k_wizardlm
    llama3_8b_budget_adjusted_v2_second_2k_1223_cleaned_5sets_euclidean_nonlinear_v4_0.95_alpha_0.3_lambda_0.9_gamma_2.0_6k_wizardlm
    llama3_8b_budget_adjusted_v2_third_2k_1223_cleaned_5sets_euclidean_nonlinear_v4_0.95_alpha_0.3_lambda_0.9_gamma_2.0_6k_wizardlm
)

for data in "${datasets[@]}"
do
    MODEL_PATH="../saves/checkpoints/${data}"

    MODEL_DIR="../saves/checkpoints/${data}"
    SAVE_DIR="../saves/alpacaeval/${data}"
    cd ../LLaMA-Factory
    torchrun src/train.py \
    --model_name_or_path "${MODEL_DIR}" \
    --stage sft \
    --do_predict \
    --finetuning_type full \
    --eval_dataset alpaca_eval \
    --template llama3 \
    --cutoff_len 2048 \
    --max_samples 1000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir "${SAVE_DIR}" \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
    --fp16 \
    --ddp_timeout 180000000 \
    --temperature 0.7 \
    --top_p 0.9 \
    --num_beams 1 \
    --max_length 2048 \
    --max_new_tokens 512 \
    --seed 8


    MODEL_ID=llama-3
    ANSWER_FILE="../saves/mtbench/llama3_template_${data}.jsonl"
    cd ../FastChat/fastchat/llm_judge
    python gen_model_answer.py --model-path $MODEL_PATH --model-id $MODEL_ID --answer-file $ANSWER_FILE --max-new-token 512
done


datasets=(
    qwen7b_budget_adjusted_v2_cleaned_5sets_kcenter_multiply_gamma_1
)

for data in "${datasets[@]}"
do
    MODEL_PATH="../saves/checkpoints/${data}"

    MODEL_DIR="../saves/checkpoints/${data}"
    SAVE_DIR="../saves/alpacaeval/${data}"
    cd ../LLaMA-Factory
    torchrun src/train.py \
    --model_name_or_path "${MODEL_DIR}" \
    --stage sft \
    --do_predict \
    --finetuning_type full \
    --eval_dataset alpaca_eval \
    --template qwen \
    --cutoff_len 2048 \
    --max_samples 1000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir "${SAVE_DIR}" \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate \
    --fp16 \
    --ddp_timeout 180000000 \
    --temperature 0.7 \
    --top_p 0.9 \
    --num_beams 1 \
    --max_length 2048 \
    --max_new_tokens 512 \
    --seed 8

    MODEL_ID=qwen-7b-chat
    ANSWER_FILE="../saves/mtbench/${data}.jsonl"
    cd ../FastChat/fastchat/llm_judge
    python gen_model_answer.py --model-path $MODEL_PATH --model-id $MODEL_ID --answer-file $ANSWER_FILE --max-new-token 512
done
