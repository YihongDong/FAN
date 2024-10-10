from datasets import load_dataset
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader, ConcatDataset

class tokenized_dataloader:
    def __init__(self, args):
        if args.model == "mamba":
            self.tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        
    def get_data_loaders(self, part_list=["train"]):
        dataset_list = []
        for part in part_list:
            if part in ["train", "validation", "test"]:
                dataset = self.get_tokenized_dataset(dataset_name=self.dataset, part=part)
                dataset_list.append(dataset)
            else:
                raise ValueError("part must be one of 'train', 'validation', 'test'")
        if len(part_list) == 1 and part_list[0] in ["validation", "test"]:
            return DataLoader(dataset_list[0], batch_size=self.batch_size, shuffle=False, num_workers=4)
        else:
            dataset = ConcatDataset(dataset_list)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
            return dataloader
        
    def get_tokenized_dataset(self, dataset_name, part="train"):
        if dataset_name == "sst2":
            dataset = load_dataset("glue", "sst2")
            dataset[part] = dataset[part].remove_columns(["idx"])
            dataset[part] = dataset[part].rename_column("sentence", "text")
        elif dataset_name == "imdb":
            dataset = load_dataset("imdb")
        elif dataset_name == "sentiment140":
            dataset = load_dataset("adilbekovich/Sentiment140Twitter", encoding='ISO-8859-1')
        elif dataset_name == "amazon_polarity":
            dataset = load_dataset("amazon_polarity")
            def combine_title_content(batch):
                batch['text'] = [title + '. ' + content for title, content in zip(batch['title'], batch['content'])]
                return batch
            dataset = dataset.map(combine_title_content, batched=True)
            dataset = dataset.remove_columns(['title', 'content'])
        
        dataset = dataset[part].map(self.tokenize_function, batched=True)
        dataset = dataset.remove_columns(['text'])
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format("torch")
        return dataset

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )