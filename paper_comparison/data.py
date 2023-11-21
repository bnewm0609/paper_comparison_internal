import json

from omegaconf import DictConfig


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f]


def load_data(args: DictConfig):
    if args.data.type == "debug_jsonl":
        return load_jsonl(args.data.path)
