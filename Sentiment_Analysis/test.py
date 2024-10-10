import os
import argparse
import torch
from model.build_model import build_model
import logging
from tqdm.auto import tqdm
import evaluate
from get_dataloader import tokenized_dataloader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def test(args):
    setup_logging(args.log_file)
    logging.info("--------------- Start testing ---------------")

    dataloader = tokenized_dataloader(args=args)
    if args.dataset == "imdb":
        part_list = ["test"]
    else:
        part_list = ["train", "test"]
    test_loader = dataloader.get_data_loaders(part_list=part_list)
    logging.info("test data on {} of {}".format(part_list, args.dataset))
    
    model = build_model(args)
    print(model)
    model_path = os.path.join('checkpoints', args.prefix + "_best.pth")
    model.load_state_dict(torch.load(model_path))

    progress_bar = tqdm(range(len(test_loader)))
    
    metric = evaluate.load("accuracy")
    # testing
    model.eval()
    total_loss = 0.0
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar.update(1)

    avg_loss = total_loss / len(test_loader)
    val_accuracy = metric.compute()['accuracy']
    
    logging.info(f"Test {args.prefix} Model on {args.dataset} part_list={part_list}")
    logging.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {val_accuracy:.4f}")

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
    test(args)
