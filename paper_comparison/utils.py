import json
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def convert_paths_to_absolute(config: DictConfig):
    """
    Turn any path in the passed config to an absolute path by recursing down the tree defined by
    the config yaml files by modifying the passed config in-place.
    """
    to_resolve = []
    for key in config:
        if isinstance(config[key], DictConfig):
            to_resolve.append(config[key])  # check if there's anything to resolve in the sub-config
        elif "path" in key:
            config[key] = to_absolute_path(config[key])
    for sub_config in to_resolve:
        convert_paths_to_absolute(sub_config)


def get_results_path(args: DictConfig) -> str:
    results_path = ""
    if args.mode == "endtoend":
        results_path = f"results/{args.mode}/{args.data._id}__etoe_{args.endtoend._id}"
    elif args.mode == "schematizer-populator":
        results_path = (
            f"results/{args.mode}/{args.data._id}__schem_{args.schematizer._id}_popu{args.populator._id}"
        )
    return results_path


def save_outputs(args: DictConfig, tables: list, metrics: dict):
    # save the tables - think about the format here - it might actually be good to save these as a
    # single, large pandas dataframe with an additional key representing the table/group of papers being
    # compared. For now, this is a jsonl file with one line per table/group of papers.
    results_path = Path(args.results_path)
    with open(results_path / "tables.jsonl", "w") as f:
        for table in tables:
            f.write(json.dumps({"tab_id": table.tabid, "table": table.values}) + "\n")

    with open(results_path / "metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Saving outputs:")
    # print("Tables:", tables)
    print("Metrics:", metrics)
