import json
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


class PrecomputedOutputsEndToEnd(BaseEndToEnd):
    def __call__(self, args, data) -> list[Table]:
        with open(args.endtoend.path) as f:
            table_values = json.load(f)

        # the baselines (baseline_paper_to_table_max.json, baseline_paper_to_cc_tab_max.json) contain only the tables
        # so, we don't need to extract them, but for Our algorithm (ours_output_decontext.json) the file contains other
        # info
        if "final_table" in table_values:
            table_values = table_values["final_table"]
            for attribute in table_values:
                del table_values[attribute]["type"]
                del table_values[attribute]["presup"]

        return [
            Table(
                tabid="0",
                schema=table_values.keys(),
                values=table_values,
            )
        ]


class OracleEndToEnd(BaseEndToEnd):
    """Returns the gold tables"""

    def __call__(self, args, data) -> list[Table]:
        return [sample["y"] for sample in data]


def load_endtoend(args: DictConfig):
    # breakpoint()
    if args.endtoend.name == "debug_abstracts":
        return DebugAbstractsEndToEnd(args)
    elif args.endtoend.name == "precomp_outputs":
        return PrecomputedOutputsEndToEnd(args)
    elif args.endtoend.name == "oracle":
        return OracleEndToEnd(args)
    return BaseEndToEnd(args)
