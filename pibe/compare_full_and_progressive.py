import os
os.environ["TQDM_DISABLE"] = "True"

import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from affinity_propagation import AffinityPropagation
from scipy.stats import spearmanr
from geomloss import SamplesLoss

import time
import json

from tqdm import tqdm
import logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.CRITICAL)

import random
random.seed(88888)


def similarity(embeddings):
    similarity_matrix = torch.tensor(cosine_similarity(embeddings))
    mask = torch.ones(similarity_matrix.shape, dtype=bool)
    mask.fill_diagonal_(0)
    filtered_similarity = similarity_matrix.masked_select(mask).view(embeddings.size(0), -1)
    top1_nn_sim = torch.topk(filtered_similarity, 1, largest=True).values.mean()

    distance_matrix = torch.tensor(euclidean_distances(embeddings))
    mask = torch.ones(distance_matrix.shape, dtype=bool)
    mask.fill_diagonal_(0)
    filtered_distance = distance_matrix.masked_select(mask).view(embeddings.size(0), -1)
    top1_nn_distance = torch.topk(filtered_distance, 1, largest=False).values.mean()

    print("**"*80)
    print(
        f"\tCos Sim: {filtered_similarity.mean()}, NN Cos Sim: {top1_nn_sim}, EU Distance: {filtered_distance.mean()}, NN EU Distance: {top1_nn_distance}"
    )

    # return filtered_similarity.mean(), top1_nn_sim, filtered_distance.mean(), top1_nn_distance,

def check_src(data):
    cnt = {
        0: 0,
        1: 0,
        2: 0, 
        3: 0
    }
    for item in data:
        cnt[item['id']] += 1
    print(f"First batch: {cnt[0]}, Second batch: {cnt[1]}, Third batch: {cnt[2]}, Fourth batch: {cnt[3]}")


def check_overlap(full_data, progressive_data):
    full_data_conv_str_bank = []
    for conversation in full_data:
        conversation = conversation['conversations']
        conv_str = ""
        for turn in conversation:
            conv_str += turn['value'] + '\n'
        conv_str = conv_str.strip()
        full_data_conv_str_bank.append(conv_str)
    full_data_conv_str_bank = set(full_data_conv_str_bank)

    progressive_data_conv_str_bank = []
    for conversation in progressive_data:
        conversation = conversation['conversations']
        conv_str = ""
        for turn in conversation:
            conv_str += turn['value'] + '\n'
        conv_str = conv_str.strip()
        progressive_data_conv_str_bank.append(conv_str)

    cnt = 0
    for turn in progressive_data_conv_str_bank:
        if turn in full_data_conv_str_bank:
            cnt += 1
    return cnt


def cal_distance_mat(data):
    embedding = torch.cat([item['embedding'].reshape(1, -1) for item in data], dim=0).cuda()
    # X_norm = embedding / embedding.norm(dim=1, keepdim=True)
    # Y_norm = embedding / embedding.norm(dim=1, keepdim=True)
    # distances = 1 - torch.mm(X_norm, Y_norm.t()).detach().cpu() / 2
    distances = torch.sqrt(torch.sum(embedding**2, dim=1, keepdim=True) - 2 * torch.mm(embedding, embedding.t()) + torch.sum(embedding**2, dim=1, keepdim=True).t()).detach().cpu()
    
    return torch.topk(distances, 2000, dim=1, largest=False)


def knn(nn_distance_mat, nn_indices_mat, quality, full_data, pool_size=2000):
    pool = []
    score, row_min_indices = torch.min(nn_distance_mat, dim=1)
    score = score.flatten()
    quality = (quality - quality.min()) / (quality.max() - quality.min())
    score = (score - score.min()) / (score.max() - score.min())
    # score = (1+score)*(1+quality)
    top_values, top_indices = torch.topk(score, pool_size)
    top_indices = top_indices.numpy().tolist()
    pool = [full_data[nn_indices_mat[i][row_min_indices[i]]] for i in top_indices]

    return pool


def progressive_knn(full_data, batch_size=10000, pool_size=2000):
    left_idx = 0
    pool = []
    while left_idx < len(full_data):
        logging.info(f"Current Left Index: {left_idx}")
        right_idx = min(left_idx + batch_size, len(full_data))
        candidate_data = full_data[left_idx:right_idx]
        cur_data = pool + candidate_data
        nn_distance_mat, nn_indices_mat = cal_distance_mat(cur_data)
        quality = [item['quality'] for item in cur_data]
        quality = torch.tensor(quality)
        quality = quality.flatten()
        pool = knn(nn_distance_mat, nn_indices_mat, quality, cur_data, pool_size)
        left_idx = right_idx

    return pool


def kcenter(nn_distance_mat, nn_indices_mat, embedding, quality, full_data, pool_size=2000):
    pool = []
    mask_idx = []
    pool_distance = None
    embedding = embedding.cuda()
    quality = (quality - quality.min()) / (quality.max() - quality.min())
    for _ in tqdm(range(pool_size)):
        if len(pool) == 0:
            row_min_values, row_min_indices = torch.min(nn_distance_mat, dim=1)
            score = 0 - row_min_values
            score = score.flatten()
            score = (score - score.min()) / (score.max() - score.min())
            score = (1+score)*(1+quality)
            max_of_min_value, max_index = torch.max(score, dim=0)
            idx = nn_indices_mat[max_index][row_min_indices[max_index]]
            pool.append(full_data[idx])
            mask_idx.append(row_min_indices[max_index])
            continue
        nn_distance_mat[:, mask_idx[-1]] = 100000  # mask already selected data

        with torch.no_grad():
            X = pool[-1]['embedding'].unsqueeze(0).cuda()
            X_norm = X / X.norm(dim=1, keepdim=True)
            Y_norm = embedding / embedding.norm(dim=1, keepdim=True)
            simlarity = torch.mm(X_norm, Y_norm.t()).detach().cpu().unsqueeze(0)
        if pool_distance is not None:
            pool_distance = torch.cat([pool_distance, simlarity], dim=0)
        else:
            pool_distance = simlarity

        row_min_values, row_min_indices = torch.min(nn_distance_mat, dim=1)
        pool_row_max_values, pool_row_max_indices = torch.max(pool_distance, dim=0)
        score = pool_row_max_values - row_min_values
        score = score.flatten()
        score = (score - score.min()) / (score.max() - score.min())
        score = (1+score) * (1+quality)
        _, index = torch.max(score, dim=0)
        idx = nn_indices_mat[index][row_min_indices[index]]
        pool.append(full_data[idx])
        mask_idx.append(row_min_indices[idx])
    return pool


def progressive_kcenter(full_data, batch_size=10000, pool_size=2000):
    left_idx = 0
    pool = []
    while left_idx < len(full_data):
        logging.info(f"Current Left Index: {left_idx}")
        right_idx = min(left_idx + batch_size, len(full_data))
        candidate_data = full_data[left_idx:right_idx]
        cur_data = pool + candidate_data
        embedding = torch.cat([item['embedding'].reshape(1, -1) for item in cur_data], dim=0).cuda()
        nn_distance_mat, nn_indices_mat = cal_distance_mat(cur_data)
        quality = [item['quality'] for item in cur_data]
        quality = torch.tensor(quality)
        pool = kcenter(nn_distance_mat, nn_indices_mat, embedding, quality, cur_data, pool_size)
        left_idx = right_idx

    return pool


def deita(full_data, pool_size=2000):
    # sorted_full_data = full_data
    # random.shuffle(sorted_full_data)
    sorted_full_data = sorted(full_data, key=lambda x: x['quality'], reverse=True)
    pool = []
    pool.append(sorted_full_data[0])
    for i in tqdm(range(1, len(sorted_full_data))):
        embedding = sorted_full_data[i]['embedding'].cuda()
        nn_distance = -1.0
        pool_embedding = torch.cat([item['embedding'].unsqueeze(0).cuda() for item in pool], dim=0)
        distance = F.cosine_similarity(embedding, pool_embedding).detach().cpu().flatten()
        if distance.max() > nn_distance:
            nn_distance = distance.max()
        if nn_distance < 0.9:
            pool.append(sorted_full_data[i])
        if len(pool) == pool_size:
            break
    return pool


def progressive_deita(full_data, batch_size=10000, pool_size=2000):
    left_idx = 0
    pool = []
    while left_idx < len(full_data):
        logging.info(f"Current Left Index: {left_idx}")
        right_idx = min(left_idx + batch_size, len(full_data))
        candidate_data = full_data[left_idx:right_idx]
        cur_data = pool + candidate_data
        pool = deita(cur_data, pool_size)
        left_idx = right_idx
    return pool


def pibe(full_data, alpha, lamb, gamma, mode, batch_size=10000, pool_size=2000):
    # cosine_similarity
    # euclidean
    ap = AffinityPropagation(affinity='euclidean', damping=0.5, batch_size=batch_size, n_clusters=pool_size, preference=0.,
                             alpha=alpha, lamb=lamb, gamma=gamma, mode=mode, save_log=False, verbose=False)
    embedding = torch.cat([item['embedding'].reshape(1, -1) for item in full_data], dim=0)
    quality = [item['quality'] for item in full_data]

    embeddings_split = []
    quality_split = []
    iters = embedding.shape[0] // batch_size
    if embedding.shape[0] % batch_size > 0:
        iters += 1
    for i in range(iters):
        right_bound = min(batch_size * i + batch_size, embedding.shape[0])
        embeddings_split.append(embedding[batch_size * i:right_bound])
        quality_split.append(quality[batch_size * i:right_bound])
    # del embedding
    del quality

    for i in tqdm(range(len(embeddings_split))):
        with torch.no_grad():
            ap = ap.incremental_fit(embeddings_split[i], quality_split[i], quality_ratio=0.)
    # similarity(embedding[ap.pool_indexes])
    pool = [full_data[i] for i in torch.tensor(ap.pool_indexes)]
    return pool


def run_knn():
    print("KNN w/o quality")
    nn_distance_mat, nn_indices_mat = cal_distance_mat(full_data)
    logging.info("Start KNN")
    full_knn_pool = knn(nn_distance_mat, nn_indices_mat, full_quality, full_data, pool_size=1000)
    logging.info("Start Progressive KNN")
    progressive_knn_pool = progressive_knn(full_data, batch_size=10000, pool_size=1000)
    check_src(full_knn_pool)
    check_src(progressive_knn_pool)
    print(f'Overlap num between KNN and Progressive KNN: {check_overlap(full_knn_pool, progressive_knn_pool)}')
    full_knn_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in full_knn_pool])
    progressive_knn_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_knn_pool])
    with torch.no_grad():
        distance = loss(full_knn_embedding.cuda(), progressive_knn_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("--"*40)


def run_pibe():
    logging.info("Start PIBE")
    full_pibe_pool = pibe(full_data, alpha=0.0, lamb=0.97, gamma=1.0, mode='multiply', batch_size=len(full_data), pool_size=1000)
    check_src(full_pibe_pool)
    full_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in full_pibe_pool])
    print("--"*40)

    logging.info("Start Progressive PIBE")

    print("Without History")
    progressive_pibe_pool = pibe(full_data, alpha=0.0, lamb=0.97, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("--"*40)
    
    progressive_pibe_pool = pibe(full_data, alpha=0.5, lamb=0.90, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.4, lamb=0.90, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.3, lamb=0.90, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.2, lamb=0.90, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.1, lamb=0.90, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("--"*40)

    progressive_pibe_pool = pibe(full_data, alpha=0.5, lamb=0.93, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.4, lamb=0.93, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.3, lamb=0.93, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.2, lamb=0.93, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.1, lamb=0.93, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("--"*40)

    progressive_pibe_pool = pibe(full_data, alpha=0.5, lamb=0.95, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.4, lamb=0.95, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.3, lamb=0.95, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.2, lamb=0.95, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.1, lamb=0.95, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("--"*40)

    # progressive_pibe_pool = pibe(full_data, alpha=0.5, lamb=0.96, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    # check_src(progressive_pibe_pool)
    # print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    # progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    # with torch.no_grad():
    #     distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
    #     print(f"Wasserstein Distance: {distance.item()}")
    # print("**"*20)

    # progressive_pibe_pool = pibe(full_data, alpha=0.4, lamb=0.96, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    # check_src(progressive_pibe_pool)
    # print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    # progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    # with torch.no_grad():
    #     distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
    #     print(f"Wasserstein Distance: {distance.item()}")
    # print("**"*20)

    # progressive_pibe_pool = pibe(full_data, alpha=0.3, lamb=0.96, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    # check_src(progressive_pibe_pool)
    # print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    # progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    # with torch.no_grad():
    #     distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
    #     print(f"Wasserstein Distance: {distance.item()}")
    # print("**"*20)

    # progressive_pibe_pool = pibe(full_data, alpha=0.2, lamb=0.96, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    # check_src(progressive_pibe_pool)
    # print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    # progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    # with torch.no_grad():
    #     distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
    #     print(f"Wasserstein Distance: {distance.item()}")
    # print("**"*20)

    # progressive_pibe_pool = pibe(full_data, alpha=0.1, lamb=0.96, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    # check_src(progressive_pibe_pool)
    # print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    # progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    # with torch.no_grad():
    #     distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
    #     print(f"Wasserstein Distance: {distance.item()}")
    # print("--"*40)

    progressive_pibe_pool = pibe(full_data, alpha=0.5, lamb=0.97, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.4, lamb=0.97, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.3, lamb=0.97, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.2, lamb=0.97, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.1, lamb=0.97, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("--"*40)

    # progressive_pibe_pool = pibe(full_data, alpha=0.5, lamb=0.98, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    # check_src(progressive_pibe_pool)
    # print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    # progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    # with torch.no_grad():
    #     distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
    #     print(f"Wasserstein Distance: {distance.item()}")
    # print("**"*20)

    # progressive_pibe_pool = pibe(full_data, alpha=0.4, lamb=0.98, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    # check_src(progressive_pibe_pool)
    # print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    # progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    # with torch.no_grad():
    #     distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
    #     print(f"Wasserstein Distance: {distance.item()}")
    # print("**"*20)

    # progressive_pibe_pool = pibe(full_data, alpha=0.3, lamb=0.98, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    # check_src(progressive_pibe_pool)
    # print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    # progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    # with torch.no_grad():
    #     distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
    #     print(f"Wasserstein Distance: {distance.item()}")
    # print("**"*20)

    # progressive_pibe_pool = pibe(full_data, alpha=0.2, lamb=0.98, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    # check_src(progressive_pibe_pool)
    # print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    # progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    # with torch.no_grad():
    #     distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
    #     print(f"Wasserstein Distance: {distance.item()}")
    # print("**"*20)

    # progressive_pibe_pool = pibe(full_data, alpha=0.1, lamb=0.98, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    # check_src(progressive_pibe_pool)
    # print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    # progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    # with torch.no_grad():
    #     distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
    #     print(f"Wasserstein Distance: {distance.item()}")
    # print("--"*40)

    progressive_pibe_pool = pibe(full_data, alpha=0.5, lamb=0.99, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.4, lamb=0.99, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.3, lamb=0.99, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.2, lamb=0.99, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("**"*20)

    progressive_pibe_pool = pibe(full_data, alpha=0.1, lamb=0.99, gamma=1.0, mode='multiply', batch_size=10000, pool_size=1000)
    check_src(progressive_pibe_pool)
    print(f'Overlap num between PIBE and Progressive PIBE: {check_overlap(full_pibe_pool, progressive_pibe_pool)}')
    progressive_pibe_embedding = torch.cat([item['embedding'].unsqueeze(0) for item in progressive_pibe_pool])
    with torch.no_grad():
        distance = loss(full_pibe_embedding.cuda(), progressive_pibe_embedding.cuda())
        print(f"Wasserstein Distance: {distance.item()}")
    print("--"*40)


def run_kcenter():
    print("K-Center w quality")
    nn_distance_mat, nn_indices_mat = cal_distance_mat(full_data)
    logging.info("Start K-Center")
    full_kcenter_pool = kcenter(nn_distance_mat, nn_indices_mat, embedding, full_quality, full_data, pool_size=1000)
    logging.info("Start Progressive K-Center")
    progressive_kcenter_pool = progressive_kcenter(full_data, batch_size=10000, pool_size=1000)
    check_src(full_kcenter_pool)
    check_src(progressive_kcenter_pool)
    print(f'Overlap num between K-Center and Progressive K-Center: {check_overlap(full_kcenter_pool, progressive_kcenter_pool)}')
    print("--"*40)


def run_deita():
    logging.info("Start Deita")
    full_deita_pool = deita(full_data, pool_size=1000)
    logging.info("Start Progressive Deita")
    progressive_deita_pool = progressive_deita(full_data, batch_size=10000, pool_size=1000)
    print(f'Overlap num between Deita and Progressive Deita: {check_overlap(full_deita_pool, progressive_deita_pool)}')
    print("--"*40)


if __name__ == "__main__":
    data_save_dir = ""

    full_data = torch.load(f'{data_save_dir}/subset_40k.pth')
    random.shuffle(full_data)
    for i in range(len(full_data)):
        full_data[i]['id'] = i//10000

    embedding = torch.cat([item['embedding'].reshape(1, -1) for item in full_data], dim=0)
    quality = [item['quality'] for item in full_data]
    quality = torch.tensor(quality)
    quality = quality.flatten()
    full_quality = (quality - quality.min()) / (quality.max() - quality.min())
    logging.info(f"Total num of candidate data: {len(full_data)}")

    loss = SamplesLoss("sinkhorn", p=2, blur=0.05)

    # run_knn()
    # run_pibe()
    run_kcenter()
    # run_deita()
