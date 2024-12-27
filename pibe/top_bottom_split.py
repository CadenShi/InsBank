import torch
import json
from tqdm import tqdm
import copy

def log(out_path):
    with open('/mnt/public02/usr/yuanpeiwen/instruction_pool_cleaned/LLaMA-Factory/data/dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    name = out_path.split('/')[-1].split('.json')[0]
    dataset_info[name] = {
        "file_name": f"{name}.json",
        "formatting": "sharegpt"
    }
    with open('/mnt/public02/usr/yuanpeiwen/instruction_pool_cleaned/LLaMA-Factory/data/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)

names = (
    # "1205_cleaned_euclidean_multiply_alpha_0.3_lambda_0.9_gamma_1.0_6k",
    # "1205_cleaned_euclidean_multiply_alpha_0.3_lambda_0.9_gamma_2.0_6k",
    # "1209_cleaned_euclidean_addition_alpha_0.3_lambda_0.9_gamma_1.0_6k",
    # "1209_cleaned_euclidean_addition_alpha_0.3_lambda_0.9_gamma_2.0_6k",
    "1223_cleaned_5sets_euclidean_nonlinear_v4_0.95_alpha_0.3_lambda_0.9_gamma_2.0_6k",
)

pairs = [
    (
        f"/mnt/public02/usr/yuanpeiwen/instruction_pool_cleaned/pool_evolve/ap_outputs/{name}/WizardLM_alpaca.pth",
        f"/mnt/public02/usr/yuanpeiwen/instruction_pool_cleaned/LLaMA-Factory/data/first_2k_{name}_wizardlm.json",
        f"/mnt/public02/usr/yuanpeiwen/instruction_pool_cleaned/LLaMA-Factory/data/second_2k_{name}_wizardlm.json",
        f"/mnt/public02/usr/yuanpeiwen/instruction_pool_cleaned/LLaMA-Factory/data/third_2k_{name}_wizardlm.json",
    ) for name in names 
]

for (data_path, top2k_path, mid2k_path, bottom2k_path) in tqdm(pairs):
    cnt = 0
    try:
        data = torch.load(data_path)
    except:
        continue
    if type(data) is dict:
        data = data['data']
    data = [{"conversations":item['conversations']} for item in data]

    samples = []
    for sample in tqdm(data):
        system_conv = []
        if sample["conversations"][0]["from"] == "system":
            system_conv.append(sample["conversations"][0])
            sample["conversations"] = sample["conversations"][1:]

        if len(sample["conversations"]) < 2:
            continue
        
        for i in range(len(sample["conversations"])):
            if sample["conversations"][i]['from'] not in ['human', 'gpt']:
                if sample["conversations"][i]['from'] == 'user':
                    sample["conversations"][i]['from'] = 'human'
                elif sample["conversations"][i]['from'] == 'system':
                    continue
                else:
                    sample["conversations"][i]['from'] = 'gpt'

        flag = False
        while sample["conversations"][0]['from'] == 'gpt' and len(sample["conversations"]) > 1:
            sample["conversations"] = sample["conversations"][1:]

        for item in sample["conversations"]:
            if item['from'] == 'gpt':
                flag = True
                break
        if len(sample["conversations"]) < 2:
            flag = False
        if flag:
            sample['conversations'] = system_conv + sample["conversations"]
            if len(system_conv) == 1:
                cnt += 1
            samples.append(sample)

    print(f"{len(samples)}, {cnt}")
    top2k_data = samples[:2000]
    mid2k_data = samples[2000:4000]
    bottom2k_data = samples[4000:6000]

    with open(top2k_path, 'w') as f:
        json.dump(top2k_data, f, indent=2, ensure_ascii=False)
        f.close()
    log(top2k_path)

    with open(mid2k_path, 'w') as f:
        json.dump(mid2k_data, f, indent=2, ensure_ascii=False)
        f.close()
    log(mid2k_path)

    with open(bottom2k_path, 'w') as f:
        json.dump(bottom2k_data, f, indent=2, ensure_ascii=False)
        f.close()
    log(bottom2k_path)
