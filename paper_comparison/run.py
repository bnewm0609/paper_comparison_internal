"""Entrypoint for running experiments."""

from datetime import datetime
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from paper_comparison.metrics import load_metrics
from paper_comparison.data import load_data
from paper_comparison.endtoend import load_endtoend
from paper_comparison.schematizer import load_schematizer
from paper_comparison.populator import load_populator
from paper_comparison.utils import save_outputs, get_results_path, convert_paths_to_absolute


def initialze_experiment(args: DictConfig):
    args.results_path = get_results_path(args)
    args.timestamp = str(datetime.now())
    ## setup seeds
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    convert_paths_to_absolute(args)

    if args.notes is not None:
        args.results_path += f"_note-{args.notes}"

    os.makedirs(args.results_path, exist_ok=True)

    # save the command that was run
    with open(os.path.join(args.results_path, "commands.txt"), "a") as f:
        f.write(" ".join(sys.argv) + "\n")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(args: DictConfig) -> None:
    """Entrypoint for running experiments.

    Registers resolvers, and sets up the experiment runner based on the passed configuration arguments.

    Args:
        args (DictConfig): The experiment configuration arguments.
    """
    OmegaConf.register_new_resolver(
        "extract_path",
        lambda path, num_segments: "-".join(os.path.splitext(path)[0].strip("/").split("/")[-num_segments:]),
    )
    print(args)
    assert args.endtoend is not None or (args.schematizer is not None and args.populator is not None)
    args.mode = "endtoend" if args.endtoend is not None else "schematizer-populator"

    initialze_experiment(args)
    data = load_data(args)
    evaluator = load_metrics(args)

    if args.mode == "endtoend":
        print("Running endtoend baseline")
        endtoend = load_endtoend(args)
        tables = endtoend(args, data)

    else:
        print("Running schematizer and populator")
        schematizer = load_schematizer(args)
        populator = load_populator(args)

        schema = schematizer(args, data)
        tables = populator(args, data, schema)

    metrics = evaluator(args, tables, data)

    save_outputs(args, tables, metrics)
    print(f"Results saved to:\n{args.results_path}")


if __name__ == "__main__":
    main()
