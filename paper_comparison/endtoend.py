from typing import Any
from omegaconf import DictConfig

from paper_comparison.types.table import Table


class BaseEndToEnd:
    def __init__(self, args: DictConfig):
        pass

    def __call__(self, args, data) -> list[Table]:
        return []


class DebugAbstractsEndToEnd(BaseEndToEnd):
    def __call__(self, args, data) -> list[Table]:
        table_values = {
            "Studies decontextualization?": {"choi21": ["yes"], "newman23": ["yes"], "potluri23": ["no"]},
            "What is their data source?": {
                "choi21": ["Wikipedia"],
                "newman23": ["Scientific Papers"],
                "potluri23": ["ELI5"],
            },
            "What field are they in?": {"choi21": ["NLP"], "newman23": ["NLP"], "potluri23": ["NLP"]},
            "What task do they study?": {
                "choi21": ["decontextualization"],
                "newman23": ["decontextualization"],
                "potluri23": ["long-answer summarization"],
            },
        }
        return [
            Table(
                tabid="0",
                schema=set(table_values.keys()),
                values=table_values,
            )
        ]


def load_endtoend(args: DictConfig):
    if args.endtoend.name == "debug_abstracts":
        return DebugAbstractsEndToEnd(args)
    return BaseEndToEnd(args)
