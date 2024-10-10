import logging
from model.CustomBERT import CustomBertClassifier
from model.Mamba import MambaForSequenceClassification
from transformers import MambaConfig


def build_model(args):
    if args.model == "trans":
        model = CustomBertClassifier(num_labels=args.num_classes, num_hidden_layers=args.num_hidden_layers, replace_ffn=args.replace_ffn, with_gate=args.with_gate).to(args.device)
        logging.info("with_gate: {}".format(args.with_gate))
    elif args.model == "mamba":
        config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
        config.num_hidden_layers = args.num_hidden_layers
        config.hidden_size = args.hidden_size
        logging.info("hidden_size: {}".format(config.hidden_size))
        logging.info("num_hidden_layers: {}".format(config.num_hidden_layers))
        model = MambaForSequenceClassification(config=config, num_labels=args.num_classes, max_pooler=args.max_pooler).to(args.device)
        print('model:', model)
    return model