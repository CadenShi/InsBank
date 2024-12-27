import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import copy

import time

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

time_begin = time.time()

sorted_full_data  = sorted(full_data, key=lambda x: x['quality'], reverse=True)

pool = []
pool.append(sorted_full_data[0])
for i in tqdm(range(1, len(sorted_full_data))):
    embedding = sorted_full_data[i]['embedding'].cuda()
    nn_distance = -1.0
    left = 0
    while True:
        right = min(left+5000, len(pool))
        pool_embedding = [pool[j]['embedding'].unsqueeze(0).cuda() for j in range(left, right)]
        pool_embedding = torch.cat(pool_embedding, dim=0)
        distance = F.cosine_similarity(embedding, pool_embedding).detach().cpu().flatten()
        if distance.max() > nn_distance:
            nn_distance = distance.max()
        left = right
        if right >= len(pool):
            break
    if nn_distance < 0.9:
        pool.append(sorted_full_data[i])

    if len(pool) % 1000 == 0:
        print(f"Collected: {len(pool)}/6000")
    if len(pool) == 6000:
        break

time_end = time.time()
elapsed_time = time_end - time_begin
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
time_cost = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

file_name = f'./deita_time_cost.txt'
with open(file_name, "w", encoding="utf-8") as file:
    file.write(time_cost)

os.makedirs("./outputs", exist_ok=True)
torch.save(pool, f'./outputs/cleaned_5sets_no_complexity_deita_6k.pth')
