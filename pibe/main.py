import numpy as np
import random
import json
import time
import copy
import torch
from tqdm import tqdm
import pandas as pd
import os
import fire
import logging

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from affinity_propagation import AffinityPropagation

from torch.cuda.amp import autocast

logging.basicConfig(level=logging.DEBUG,  
                    format='%(asctime)s - %(levelname)s - %(message)s')  

data_save_dir = ""

datasets =[
    {'name': 'Self_instruct', 'path': f'{data_save_dir}/cleaned_self_instruction_instruction_embedding.pth'},
    {'name': 'Alpaca_gpt4', 'path': f'{data_save_dir}/cleaned_alpaca_gpt4_data_instruction_embedding.pth'},
    {'name': 'Dolly', 'path': f'{data_save_dir}/cleaned_databricks_dolly_15k_instruction_embedding.pth'},
    {'name': 'ShareGPT', 'path': f'{data_save_dir}/cleaned_ShareGPT_V3_unfiltered_cleaned_split_no_imsorry_instruction_embedding.pth'},
    {'name': 'WizardLM_alpaca', 'path': f'{data_save_dir}/cleaned_alpaca_evol_instruct_70k_instruction_embedding.pth'},
]


def main(
    n_clusters: int=6000,
    batch_size: int=27000,
    affinity: str='euclidean',
    damping: float=0.5,
    alpha: float=0.3,
    lamb: float=0.9,
    gamma: float=0.0,
    mode: str='multiply',
):
    print(
        f"Progressive InsBank Evolution with Params: \n"
        f"n_clusters: {n_clusters}\n"
        f"batch_size: {batch_size}\n"
        f"affinity: {affinity}\n"
        f"damping: {damping}\n"
        f"alpha: {alpha}\n"
        f"lamb: {lamb}\n"
        f"gamma: {gamma}\n"
        f"mode: {mode}\n"
    )

    ap=AffinityPropagation(affinity=affinity, damping=damping, batch_size=batch_size, n_clusters=n_clusters, preference=0., alpha=alpha, lamb=lamb, gamma=gamma, mode=mode)
    ap_results = []
    all_full = []
    time_cost_dict = {}
    for item in datasets:
        time_begin = time.time()
        name = item['name']
        path = item['path']

        logging.info(f"Processing: {name}\n")
        load_data = torch.load(path)
        all_full += copy.deepcopy(load_data)
        embedding = torch.cat([item['embedding'].reshape(1, -1) for item in load_data], dim=0)
        quality = [item['quality'] for item in load_data]
        del load_data
        
        embeddings_split = []
        quality_split = []
        iters = embedding.shape[0]//batch_size
        if embedding.shape[0]%batch_size > 0:
            iters += 1
        for i in range(iters):
            right_bound = min(batch_size*i+batch_size, embedding.shape[0])
            embeddings_split.append(embedding[batch_size*i:right_bound])
            quality_split.append(quality[batch_size*i:right_bound])
        
        del embedding
        del quality
        for i in tqdm(range(len(embeddings_split))):
            with torch.no_grad():
                with autocast():
                    ap = ap.incremental_fit(embeddings_split[i], quality_split[i])

        time_end = time.time()
        elapsed_time = time_end - time_begin
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_cost_dict[name] = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        ap_results.append(
            {
                'name': name,
                'indexes': torch.tensor(ap.pool_indexes),
                'representative_scores': ap.representative_scores,
                'quality_scores': ap.quality_scores,
                'overall_scores': ap.overall_scores,
            }
        )
        pool_data = [all_full[i] for i in ap_results[-1]['indexes']]
        ap_results[-1]['data'] = pool_data
        num = n_clusters // 1000
        os.makedirs(f'./ap_outputs/1223_cleaned_5sets_{affinity}_{mode}_alpha_{alpha}_lambda_{lamb}_gamma_{gamma}_{num}k', exist_ok=True)
        torch.save(ap_results[-1], f'./ap_outputs/1223_cleaned_5sets_{affinity}_{mode}_alpha_{alpha}_lambda_{lamb}_gamma_{gamma}_{num}k/{name}.pth')
    with open(f'./ap_outputs/1223_cleaned_5sets_{affinity}_{mode}_alpha_{alpha}_lambda_{lamb}_gamma_{gamma}_{num}k/time_cost.json', 'w') as f:
        json.dump(time_cost_dict, f, indent=2)
        f.close()


if __name__ == '__main__':
    fire.Fire(main)

