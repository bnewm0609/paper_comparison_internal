from typing import Any
from omegaconf import DictConfig


class BaseSchematizer:
    def __init__(self, args: DictConfig):
        pass

    def __call__(self, args, data) -> Any:
        return "Schema"


def load_schematizer(args: DictConfig):
    return BaseSchematizer(args)
