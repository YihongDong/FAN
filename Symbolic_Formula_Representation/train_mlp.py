import torch, os, argparse, json
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from kan import LBFGS


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP')
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


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.depth = depth

        layers = []
        
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def get_param_size(self):
        total_parameters = 0
        total_parameters += (self.input_size + 1) * self.hidden_size
        for _ in range(self.depth - 1):
            total_parameters += (self.hidden_size + 1) * self.hidden_size
        total_parameters += (self.hidden_size + 1) * self.output_size
        return total_parameters


def train_with_test(model, dataset, ckpt_dir):

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    criterion = nn.MSELoss()
    optimizer = LBFGS(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=0.01,
                      history_size=10, 
                      line_search_fn="strong_wolfe", 
                      tolerance_grad=1e-32, 
                      tolerance_change=1e-32, 
                      tolerance_ys=1e-32)
    
    model.train()
    for _ in tqdm(range(1800)):
        def closure():
            optimizer.zero_grad()
            output = model(dataset['train_input'])
            loss = criterion(output, dataset['train_label'])
            loss.backward()
            return loss
        optimizer.step(closure)
    
    torch.save(model.state_dict(), f'{ckpt_dir}/model.pth')

    model.eval()
    with torch.no_grad():
        output = model(dataset['test_input'])
        test_loss = criterion(output, dataset['test_label']).item()
    return test_loss


if __name__ == '__main__':
    args = parse_args()
    if args.dataset_idx == 0:
        dataset = load_dataset(args, 0)
        input_size, output_size = 1, 1
    elif args.dataset_idx == 1:
        dataset = load_dataset(args, 1)
        input_size, output_size = 2, 1
    elif args.dataset_idx == 2:
        dataset = load_dataset(args, 2)
        input_size, output_size = 2, 1
    elif args.dataset_idx == 3:
        dataset = load_dataset(args, 3)
        input_size, output_size = 4, 1
    
    save_dir = f'{args.save_dir}/dataset_{args.dataset_idx}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = open(f'{save_dir}/results.jsonl', 'w')
    
    for depth in [2, 3, 4, 5]:
        for hidden_size in [2, 4, 8, 16, 32, 64, 128]:
            print(f'Depth: {depth}, Hidden size: {hidden_size}')
            model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size, depth=depth).to(device)
            param_size = sum(p.numel() for p in model.parameters())
            ckpt_dir = f'{save_dir}/depth_{depth}_hidden_{hidden_size}'
            test_loss = train_with_test(model, dataset, ckpt_dir)

            output_js = {}
            output_js['depth'] = depth
            output_js['hidden_size'] = hidden_size
            param_size = model.get_param_size()
            output_js['param_size'] = param_size
            output_js['test_loss'] = test_loss
            log_file.write(json.dumps(output_js) + '\n')
            log_file.flush()
    log_file.close()
