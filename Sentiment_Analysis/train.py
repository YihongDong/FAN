import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from model.build_model import build_model
import logging
from transformers import BertTokenizer, AutoTokenizer, DataCollatorWithPadding, BertConfig, MambaConfig
from transformers import get_scheduler
from tqdm.auto import tqdm
from functools import partial
import evaluate
from utils import view_params
from get_dataloader import tokenized_dataloader


def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples['sentence'],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

def train(args):
    torch.manual_seed(42)
    setup_logging(args.log_file)
    logging.info("--------------- Start training ---------------")
    
    dataloader = tokenized_dataloader(args=args)
    train_loader = dataloader.get_data_loaders(part_list=['train'])
    val_loader = dataloader.get_data_loaders(part_list=['validation'])
    
    logging.info("the model is: {}".format(args.model))
    
    model = build_model(args)
    
    params = view_params(model)
    logging.info(params)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    logging.info("Optimizer: {}".format(optimizer))
    logging.info(f"Learning rate: {args.learning_rate}")
    
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="cosine_with_restarts",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    best_val_accuracy = 0.0
    progress_bar = tqdm(range(num_training_steps))
    
    for epoch in range(args.epochs):
        model.train()
        metric = evaluate.load("accuracy")
        
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            if args.model == 'linear':
                predictions = (logits > 0.5).long()
            else:
                predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        avg_loss = total_loss / len(train_loader)
        train_accuracy = metric.compute()['accuracy']
        logging.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        
        # validation
        model.eval()
        total_loss = 0.0
        for batch in val_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            logits = outputs.logits
            if args.model == 'linear':
                predictions = (logits > 0.5).long()
            else:
                predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        avg_loss = total_loss / len(val_loader)
        val_accuracy = metric.compute()['accuracy']
        logging.info(f"Epoch {epoch + 1}, Validation Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
         # save model with best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_directory = f"checkpoints/{args.prefix}_best.pth"
            torch.save(model.state_dict(), save_directory)
            logging.info("Save model with best validation accuracy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--replace_ffn", action='store_true')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log_file", type=str, default="training.log")
    parser.add_argument("--prefix", type=str, default="baseline")
    parser.add_argument("--model", type=str, default="trans")
    # for FAN layer
    parser.add_argument("--with_gate", action='store_true')
    
    # for trans and mamba
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=768)
    
    # form mamba
    parser.add_argument("--max_pooler", action='store_true')
    
    # for dataset
    parser.add_argument("--dataset", type=str, default="sst2")

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    train(args)