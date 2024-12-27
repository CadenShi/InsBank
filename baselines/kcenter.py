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
embedding = torch.cat([self_instruct, alpaca_data, dolly, alpaca_evol]).cuda()
for i in range(len(full_data)):
    full_data[i]['idx'] = i

quality = [item['quality'] for item in full_data]
quality = torch.tensor(quality)
quality = quality.flatten()
quality = (quality - quality.min()) / (quality.max() - quality.min())

# min_values = []
# min_indices = []
# for left in tqdm(range(0, len(full_data), 3000)):
#     right = min(left+3000, len(full_data))
#     embedding_for_cal = embedding[left:right]
#     with torch.no_grad():
#         # X_norm = embedding_for_cal / embedding_for_cal.norm(dim=1, keepdim=True)
#         # Y_norm = embedding / embedding.norm(dim=1, keepdim=True)
#         # cosine_distance_part = 1-torch.mm(X_norm, Y_norm.t()).detach().cpu()
#         distances_part = torch.cdist(embedding_for_cal, embedding, p=2).detach().cpu()

#     values, indices = torch.topk(distances_part, 6000, dim=1, largest=False)
#     min_values.append(values)
#     min_indices.append(indices)

# nn_distance_mat = torch.cat(min_values, dim=0)
# nn_indices_mat = torch.cat(min_indices, dim=0)

# result = {
#     'distance_mat': nn_distance_mat,
#     'min_indices_mat': nn_indices_mat,
# }
# torch.save(result, './cleaned_4sets_top_6k_nn_distance_mat.pth')

result = torch.load('./cleaned_5sets_top_6k_nn_distance_mat.pth')
nn_distance_mat = result['distance_mat']
nn_indices_mat = result['min_indices_mat']

pool = []
mask_idx = []
pool_distance = None
for i in tqdm(range(6000)):
    if len(pool) == 0:
        row_min_values, row_min_indices = torch.min(nn_distance_mat, dim=1)
        score = 0 - row_min_values
        score = score.flatten()
        score = (score - score.min()) / (score.max() - score.min())
        score = (1+score)*(1+quality).pow(1)
        # score = score + 1*quality
        max_of_min_value, max_index = torch.max(score, dim=0)
        idx = nn_indices_mat[max_index][row_min_indices[max_index]]
        pool.append(full_data[idx])
        mask_idx.append(row_min_indices[max_index])
        continue

    nn_distance_mat[:, mask_idx[-1]] = 100000  # mask already selected data

    with torch.no_grad():
        X = pool[-1]['embedding'].unsqueeze(0).cuda()
        # X_norm =  X / X.norm(dim=1, keepdim=True)
        # Y_norm = embedding / embedding.norm(dim=1, keepdim=True)
        # simlarity = torch.mm(X_norm, Y_norm.t()).detach().cpu().unsqueeze(0)
        distances_part = torch.cdist(X, embedding, p=2).detach().cpu()
    if pool_distance is not None:
        pool_distance = torch.cat([pool_distance, distances_part], dim=0)
    else:
        pool_distance = distances_part

    row_min_values, row_min_indices = torch.min(nn_distance_mat, dim=1)
    pool_row_min_values, pool_row_min_indices = torch.min(pool_distance, dim=0)
    score = pool_row_min_values - row_min_values
    score = score.flatten()
    score = (score - score.min()) / (score.max() - score.min())
    score = (1+score)*(1+quality).pow(1)
    # score = score + 1*quality
    _, index = torch.max(score, dim=0)
    idx = nn_indices_mat[index][row_min_indices[index]]
    pool.append(full_data[idx])
    mask_idx.append(row_min_indices[idx])

os.makedirs("./outputs", exist_ok=True)
torch.save(pool, f'./outputs/cleaned_5sets_kcenter_addition_gamma_1.pth')
