import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='dataset', default='MNIST')
parser.add_argument('--gpu_id', type=int, help='gpu_id', default=1)
parser.add_argument('--lr', type=float, help='lr', default=0.01)
parser.add_argument('--epoch', type=int, help='epoch', default=100)
parser.add_argument('--version', type=str, help='version', default='fan')
parser.add_argument('--similarparameter', type=bool, help='similarparameter', default=True)

args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2023)  


from FANLayer import FANLayer

class CNNModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=10):
        super(CNNModel, self).__init__()
        self.conv_layer = nn.Sequential(

            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.scalar = lambda x: x*4//3 if args.similarparameter else x
        
        if args.version == 'mlp':
            self.fc_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 7 * 7, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 10)
            )
        else:
            self.fc_layer = nn.Sequential(
                nn.Flatten(),
                FANLayer(128 * 7 * 7, self.scalar(256)), #nn.Linear(128 * 7 * 7, 256),
                nn.BatchNorm1d(self.scalar(256)),
                nn.Dropout(0.5),
                nn.Linear(self.scalar(256), output_dim)
            )
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x


device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

model = CNNModel()
model.to(device)

num_epochs = args.epoch

def run(model, train_loader, OOD_test_loader, test_loader, num_epochs, name):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    best_accuracy = [0.0, 0.0]

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        testloss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                testloss += loss.item() * images.size(0)

        epoch_accuracy = 100 * correct / total
        epoch_test_loss = testloss / len(test_loader.dataset)

        if OOD_test_loader is not None:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in OOD_test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

            OOD_accuracy = 100 * correct / total
            
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_test_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, OOD Accuracy: {OOD_accuracy:.2f}%')
        
        scheduler.step()

        if epoch_accuracy > best_accuracy[0]:
            best_accuracy = [epoch_accuracy, best_accuracy[1]]
        if OOD_accuracy > best_accuracy[1]:
            best_accuracy = [best_accuracy[0], OOD_accuracy]
        
    return {'best_accuracy': best_accuracy[0], 'best_OOD_accuracy': best_accuracy[1],\
            'accuracy': epoch_accuracy, 'OOD_accuracy': OOD_accuracy}


def get_dataloader(dataset, batch_size=256, shuffle=True, Train=True):
        def transform_m(example):
            example['image'] = TF.resize(example['image'], (28, 28))
            example['image'] = example['image'].convert('L')
            example['image'] = TF.to_tensor(example['image'])
            example['image'] = TF.normalize(example['image'], mean=(.5,), std=(.5,))
            return example
        

        def collate_fn(batch):
            images = [torch.tensor(item['image']) for item in batch if not isinstance(item['image'], torch.Tensor)]
            labels = [torch.tensor(item['label']) for item in batch if not isinstance(item['label'], torch.Tensor)]
            
            images = torch.stack(images, dim=0)
            labels = torch.stack(labels, dim=0)
            
            return images, labels

        if Train: 
            trainset = dataset['train'].map(transform_m)
            testset = dataset['test'].map(transform_m)
            
            train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
            test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            return train_loader, test_loader
        else:
            testset = dataset['test'].map(transform_m)
            
            test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            return test_loader


transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

if args.dataset == 'MNIST':

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
    
    OOD_test_loader = get_dataloader(load_dataset("Mike0307/MNIST-M"), Train=False)

    accuracy_checkpoints = run(model, train_loader, OOD_test_loader, test_loader, num_epochs, name='mnist')

elif args.dataset == 'MNIST-M':
    dataset = load_dataset("Mike0307/MNIST-M")

    train_loader, test_loader = get_dataloader(dataset)
    
    OOD_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    OOD_test_loader = DataLoader(dataset=OOD_dataset, batch_size=256, shuffle=False)

    accuracy_checkpoints = run(model, train_loader, OOD_test_loader, test_loader, num_epochs, name='m_mnist')

elif args.dataset == 'Fashion-MNIST':
    dataset = load_dataset("zalando-datasets/fashion_mnist")
    train_loader, test_loader = get_dataloader(dataset)
    
    OOD_test_loader = get_dataloader(load_dataset("mweiss/fashion_mnist_corrupted"), Train=False)

    accuracy_checkpoints = run(model, train_loader, OOD_test_loader, test_loader, num_epochs, name='f_mnist')

elif args.dataset == 'Fashion-MNIST-corrupted':
    dataset = load_dataset("mweiss/fashion_mnist_corrupted")
    train_loader, test_loader = get_dataloader(dataset)

    OOD_test_loader = get_dataloader(load_dataset("zalando-datasets/fashion_mnist"), Train=False)

    accuracy_checkpoints = run(model, train_loader, OOD_test_loader, test_loader, num_epochs, name='fc_mnist')


print(f'{args.dataset}:', accuracy_checkpoints)
