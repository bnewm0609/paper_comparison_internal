from typing import Any
from omegaconf import DictConfig

from paper_comparison.types.table import Table


class BaseEndToEnd:
    def __init__(self, args: DictConfig):
        pass

    def __call__(self, args, data) -> dict[str, Table]:
        return {"": Table(set(), dict())}


class DebugAbstractsEndToEnd(BaseEndToEnd):
    def __call__(self, args, data) -> dict[str, Table]:
        table_values = {
            "Studies decontextualization?": {"Choi21": ["yes"], "Newman23": ["yes"], "Potluri23": ["no"]},
            "What is their data source?": {
                "Choi21": ["Wikipedia"],
                "Newman23": ["Scientific Papers"],
                "Potluri23": ["ELI5"],
            },
            "What field are they in?": {"Choi21": ["NLP"], "Newman23": ["NLP"], "Potluri23": ["NLP"]},
            "What task do they study?": {
                "Choi21": ["decontextualization"],
                "Newman23": ["decontextualization"],
                "Potluri23": ["long-answer summarization"],
            },
        }
        return {
            "debug": Table(
                schema=set(table_values.keys()),
                values=table_values,
            )
        }


def load_endtoend(args: DictConfig):
    if args.endtoend.name == "debug_abstracts":
        return DebugAbstractsEndToEnd(args)
    return BaseEndToEnd(args)
