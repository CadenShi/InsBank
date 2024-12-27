import os
import sys

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

import numpy as np
import random
import json
import time
import copy
import torch
from tqdm import tqdm
import pandas as pd
import fire
import logging

from affinity_propagation import AffinityPropagation

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
# from SKLEARN.cluster import AffinityPropagation

logging.basicConfig(level=logging.DEBUG,  # 设置最低日志级别为DEBUG
                    format='%(asctime)s - %(levelname)s - %(message)s')  # 设置日志格式

datasets =[
    {'name': 'quality_controlled_v5', 'path': '/mnt/public02/usr/yuanpeiwen/instruction_pool_cleaned/analysis/quality_controlled_data_v5.pth'},
    #quality ranges from 4.5 to 5.0
]


def main(
    n_clusters: int=19805,
    batch_size: int=19805,
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

    ap=AffinityPropagation(affinity=affinity, damping=damping, batch_size=batch_size, n_clusters=n_clusters, preference=0., alpha=alpha, lamb=lamb, gamma=gamma, mode=mode, save_log=False)
    ap_results = []
    all_full = []
    time_cost_dict = {}
    for item in datasets:
        time_begin = time.time()
        name = item['name']
        path = item['path']

        logging.info(f"Processing: {name}\n")
        load_data = orch.load(path)
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
        os.makedirs(f'./ap_outputs/quality_control_v5_{affinity}_{mode}_alpha_{alpha}_lambda_{lamb}_gamma_{gamma}', exist_ok=True)
        torch.save(ap_results[-1], f'./ap_outputs/quality_control_v5_{affinity}_{mode}_alpha_{alpha}_lambda_{lamb}_gamma_{gamma}/{name}.pth')
    with open(f'./ap_outputs/quality_control_v5_{affinity}_{mode}_alpha_{alpha}_lambda_{lamb}_gamma_{gamma}/time_cost.json', 'w') as f:
        json.dump(time_cost_dict, f, indent=2)
        f.close()


if __name__ == '__main__':
    fire.Fire(main)
