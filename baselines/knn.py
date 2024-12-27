import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import time

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import numpy as np


data_save_dir = ""

self_instruct_full = torch.load(f'{data_save_dir}/cleaned_self_instruction_instruction_embedding.pth')
self_instruct = torch.cat([item['embedding'].reshape(1, -1) for item in self_instruct_full], dim=0)

alpaca_data_full = torch.load(f'{data_save_dir}/cleaned_alpaca_gpt4_data_instruction_embedding.pth')
alpaca_data = torch.cat([item['embedding'].reshape(1, -1) for item in alpaca_data_full], dim=0)

alpaca_evol_full = torch.load(f'{data_save_dir}/cleaned_alpaca_evol_instruct_70k_instruction_embedding.pth')
alpaca_evol = torch.cat([item['embedding'].reshape(1, -1) for item in alpaca_evol_full], dim=0)

dolly_full = torch.load(f'{data_save_dir}/cleaned_databricks_dolly_15k_instruction_embedding.pth')
dolly = torch.cat([item['embedding'].reshape(1, -1) for item in dolly_full], dim=0)

sharegpt_full = torch.load(f'{data_save_dir}/cleaned_ShareGPT_V3_unfiltered_cleaned_split_no_imsorry_instruction_embedding.pth')
sharegpt = torch.cat([item['embedding'].reshape(1, -1) for item in sharegpt_full], dim=0)

full_data = self_instruct_full + alpaca_data_full + dolly_full + sharegpt_full + alpaca_evol_full
for i in range(len(full_data)):
    full_data[i]['idx'] = i


quality = [item['quality'] for item in full_data]
quality = torch.tensor(quality)
quality = quality.flatten()
quality = (quality - quality.min()) / (quality.max() - quality.min())

result = torch.load('./cleaned_5sets_top_6k_nn_distance_mat.pth')
nn_similarity_mat = result['distance_mat']
nn_indices_mat = result['min_indices_mat']

pool = []
score, row_min_indices = torch.min(nn_similarity_mat, dim=1)
# score = 0 - row_max_values
score = score.flatten()
diversity = score[:]
score = (score - score.min()) / (score.max() - score.min())
# score = (0.5*score + quality).flatten()
score = (1+score)*(1+quality).pow(1)
# score = score + 2*quality
top_values, top_indices = torch.topk(score, 6000)
top_indices = top_indices.numpy().tolist()
pool = [full_data[nn_indices_mat[i][row_min_indices[i]]] for i in top_indices]

print(spearmanr(score, quality))
print(spearmanr(score, diversity))

quality = [item['quality'] for item in full_data]
quality = torch.tensor(quality)
pool = {
    'data': pool,
    'diversity': diversity,
    'quality': quality
}

os.makedirs("./outputs", exist_ok=True)
torch.save(pool, f'./outputs/cleaned_5sets_knn_multiply_gamma_1.pth')
