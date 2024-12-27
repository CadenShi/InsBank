# InsBank: Evolving Instruction Subset for Ongoing Alignment

This is the introduction to code for **InsBank: Evolving Instruction Subset for Ongoing Alignment**.

# Datasets used by InsBank

- Self-Instruct https://huggingface.co/datasets/yizhongw/self_instruct

- Alpaca https://huggingface.co/datasets/yahma/alpaca-cleaned

- Dolly https://huggingface.co/datasets/databricks/databricks-dolly-15k

- WizardLM(Alpaca) https://huggingface.co/datasets/cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered

- ShareGPT https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered

# Data Quality Annotator

https://huggingface.co/hkust-nlp/deita-quality-scorer

# Run Code

## Environment

1. Follow the instructions to install [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory/tree/main) to run sft and prediction
2. Follow the instructions to install [**Fastchat**](https://github.com/lm-sys/FastChat/tree/main)  to run MT-Bench prediction and evaluation
3. Follow the instructions to install [**AlpacaEval**](https://github.com/tatsu-lab/alpaca_eval)  to run AlpacaEval evaluation
4. Follow the instructions to install [**lm-evaluation-harness**](https://github.com/EleutherAI/lm-evaluation-harness) to run IFEval prediction

⚠️ There may be conflicts between different libraries. AlpacaEval, MT-Bench, and LLaMA-Factory are compatible with each other, but lm-evaluation-harness used with vllm can conflict with LLaMA-Factory. We construct a separate environment for lm-evaluation-harness.

## Run Scrips

scrips/ \
|-- ifeval.sh \
|-- predict.sh \
|-- pibe.sh \
`-- train_sft.sh

1. `pibe.sh`: To run pibe - our data selection process
2. `train_sft`: To run instruction tuning
3. `predict.sh`: To run model prediction for AlpacaEval and Prediction. The [**AlpacaEval dataset**](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval.json) should be added to llamafactory before run this scrip. Please refer to [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory/tree/main) for checking the details about how to use your custom dataset.
4. `ifeval.sh`: Run run ifeval evaluation
5. For MT-Bench annotation and AlpacaEVal annotation, please refer to the instruction of [**MT-Bench**](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) and [**AlpacaEval**](https://github.com/tatsu-lab/alpaca_eval).
