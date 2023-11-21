from typing import Any
from omegaconf import DictConfig


class BasePopulator:
    def __init__(self, args: DictConfig):
        pass

    def __call__(self, args, data, schema) -> Any:
        return f"Table from Populator with schema: {schema}"


def load_populator(args: DictConfig):
    return BasePopulator(args)
