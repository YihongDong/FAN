from kan import *
import torch
import argparse, json
import numpy as np
import pdb
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Train KAN')
    parser.add_argument('--dataset_idx', type=int, default=0, help='Dataset index')
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--save_dir', type=str, default='kan_checkpoint')
    return parser.parse_args()


def load_dataset(args, dataset_idx):
    print(f'Loading dataset_{dataset_idx} from {args.dataset_dir}/dataset_{dataset_idx}.pt')

    dataset = torch.load(f'{args.dataset_dir}/dataset_{dataset_idx}.pt')
    dataset['train_input'] = dataset['train_input'].to(device)
    dataset['test_input'] = dataset['test_input'].to(device)
    dataset['train_label'] = dataset['train_label'].to(device)
    dataset['test_label'] = dataset['test_label'].to(device)
    return dataset


def compute_kan_size(width, grid, k):
    kan_size = 0
    for i in range(len(width) - 1):
        kan_size += (width[i][0] * width[i+1][0] * (grid + k + 3) + width[i+1][0])
    return kan_size


if __name__ == '__main__':
    args = parse_args()
    if args.dataset_idx == 0:
        dataset = load_dataset(args, 0)
        width = [1, 1]
    elif args.dataset_idx == 1:
        dataset = load_dataset(args, 1)
        width = [2, 1, 1]
    elif args.dataset_idx == 2:
        dataset = load_dataset(args, 2)
        width = [2, 2, 1]
    elif args.dataset_idx == 3:
        dataset = load_dataset(args, 3)
        width = [4, 4, 2, 1]
    else:
        raise ValueError('Invalid dataset index')
    
    save_dir = f'{args.save_dir}/dataset_{args.dataset_idx}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = open(f'{save_dir}/results.jsonl', 'a')
    
    grids = [3, 5, 10, 20, 50, 100, 200, 500, 1000]
    for i, grid in enumerate(grids):
        if i == 0:
            ckpt_dir = f'{save_dir}/ckpt'
            model = KAN(width=width, grid=grid, k=3, device=device, ckpt_path=ckpt_dir)
        else:
            model = model.refine(grid)
        results = model.fit(dataset, opt="LBFGS", steps=200, lr=0.01)

        output_js = {}
        output_js['grid'] = grid
        param_size = compute_kan_size(width, grid, 3)
        output_js['param_size'] = param_size
        output_js['train_loss'] = results['train_loss'][-1].item()
        output_js['test_loss'] = results['test_loss'][-1].item()
        log_file.write(json.dumps(output_js) + '\n')
        log_file.flush()
    log_file.close()