#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

cd ../pool_evolve

N_CLUSTERS=6000
BATCH_SIZE=27000
DAMPING=0.5
ALPHA=0.3
LAMB=0.9
GAMMA_PARAM=1
AFFINITY=euclidean      # [cosine_similarity, euclidean]
MODE=multiply           # [multiply, addition, nonlinear]

python main.py \
    --n_clusters $N_CLUSTERS\
    --batch_size $BATCH_SIZE\
    --affinity $AFFINITY\
    --damping $DAMPING\
    --alpha $ALPHA\
    --lamb $LAMB\
    --gamma $GAMMA_PARAM\
    --mode $MODE

