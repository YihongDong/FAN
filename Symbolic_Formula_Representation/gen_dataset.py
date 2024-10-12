# The Code of this part is based on KAN (https://github.com/KindXiaoming/pykan).

import torch
import scipy.special
import numpy as np
import json
import os
from tqdm import tqdm
from kan import *


device = torch.device('cpu')

def produce_dataset(dataset_idx):

    if dataset_idx == 0:
        f = lambda x: torch.tensor(torch.special.bessel_j0(20 * x[:, [0]]))
        dataset = create_dataset(f, n_var=1, train_num=3000, device=device)
    elif dataset_idx == 1:
        def f(x):
            return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [0]]**2)
        dataset = create_dataset(f, n_var=2, train_num=3000, device=device)
    elif dataset_idx == 2:
        f = lambda x: x[:, [0]] * x[:, [1]]
        dataset = create_dataset(f, n_var=2, train_num=3000, device=device)
    elif dataset_idx == 3:
        f = lambda x: torch.exp((torch.sin(torch.pi*(x[:,[0]]**2+x[:,[1]]**2))+torch.sin(torch.pi*(x[:,[2]]**2+x[:,[3]]**2)))/2)
        dataset = create_dataset(f, n_var=4, train_num=3000, device=device)
    return dataset


if __name__ == '__main__':
    save_dir = 'dataset'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(4):
        if i == 1:
            continue
        dataset = produce_dataset(i)
        torch.save(dataset, f'{save_dir}/dataset_{i}.pt')
        print(f'dataset_{i} saved into {save_dir}/dataset_{i}.pt')
