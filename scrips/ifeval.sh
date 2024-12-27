#!/bin/bash

# cd ../LLaMA-Factory
# pip install .
# pip install deepspeed==0.14.4 accelerate==1.0.1 transformers==4.42.1

cd ../lm-evaluation-harness
pip install -e ".[math,ifeval,sentencepiece]"
pip install langdetect immutabledict 
pip install nltk==3.9.1
pip install vllm


datasets=(
    robust_mistral7b_cleaned_full
)

cd ../lm-evaluation-harness/benchmarks
for data in "${datasets[@]}"
do
    MODEL_PATH="../saves/checkpoints/${data}"

    lm_eval \
    --model vllm \
    --tasks leaderboard_ifeval \
    --model_args "pretrained=../saves/checkpoints/${data},dtype=float16" \
    --batch_size 16 \
    --show_config \
    --gen_kwargs 'temperature=0' \
    --seed 0 \
    --output_path ../saves/openllm/leaderboard_ifeval \
    --trust_remote_code \
    --log_samples

done