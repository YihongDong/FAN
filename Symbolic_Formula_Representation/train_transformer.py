import torch, os, argparse, json
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from kan import LBFGS
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pdb


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


class TransformerRegressor(nn.Module):
    def __init__(self, model_dim=64, num_layers=2):
        super(TransformerRegressor, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(1, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=4, dim_feedforward=4*model_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, 1)
        
    def forward(self, x):
        x = x.unsqueeze(2)  
        x = self.embedding(x)  
        x = x.permute(1, 0, 2)  
        x = self.transformer_encoder(x)  
        x = x.mean(dim=0)  
        x = self.fc_out(x)  
        return x
    
    def get_param_size(self):
        return sum(p.numel() for p in self.parameters())

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
    

    for layer_num in [2, 3, 4, 5]:
        for dim in [4, 8, 12, 16]:
            print(f'Layer Num: {layer_num}, Model Dim: {dim}, FFN Dim: {4*dim}')
            model = TransformerRegressor(dim, layer_num).to(device)
            param_size = model.get_param_size()
            ckpt_dir = f'{save_dir}/depth_{layer_num}_hidden_{dim}'
            test_loss = train_with_test(model, dataset, ckpt_dir)

            output_js = {}
            output_js['depth'] = layer_num
            output_js['hidden_size'] = dim
            param_size = model.get_param_size()
            output_js['param_size'] = param_size
            output_js['test_loss'] = test_loss
            log_file.write(json.dumps(output_js) + '\n')
            log_file.flush()
    log_file.close()
