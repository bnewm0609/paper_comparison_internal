from typing import Any
from omegaconf import DictConfig


class BaseEval:
    def __init__(self, args) -> None:
        pass

    def __call__(self, args, tables, data) -> Any:
        return {"eval": "evals_here"}


def load_eval(args: DictConfig):
    return BaseEval(args)
