"""Entrypoint for running experiments."""

from datetime import datetime
import os
import sys
import random
import json

import hydra
from omegaconf import DictConfig, OmegaConf

from paper_comparison.metrics import load_metrics
from paper_comparison.data import load_data
from paper_comparison.endtoend import load_endtoend
from paper_comparison.utils import save_outputs, get_results_path, convert_paths_to_absolute, load_outputs, load_experiment_setting, save_experiment_setting


def initialze_experiment(args: DictConfig):
    args.results_path = get_results_path(args)
    print(args.results_path)
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
    # assert args.endtoend is not None or (args.schematizer is not None and args.populator is not None)
    args.mode = "baseline" if args.endtoend.name == "baseline_outputs" else "ours"
    
    initialze_experiment(args)
    data = load_data(args)
    evaluator = load_metrics(args) 
    print(f"Running endtoend {args.mode}")
    endtoend = load_endtoend(args)
    print(f"there are {len(data)} number of papers\n")
    retry_num = 10
    # tables = load_outputs(args)

    if not os.path.exists(f"{args.results_path}/errors"):
        os.makedirs(f"{args.results_path}/errors", exist_ok=True)
    error_save_path = f"{args.results_path}/errors/{args.difficulty}_{args.endtoend.model_type}_{args.endtoend.name}_{args.retry_type}.json"
    error_data = []
    if args.handle_error and os.path.exists(error_save_path):
        with open(error_save_path, "r") as f:
            error_data = json.load(f)

    for index, paper in enumerate(data):
        tab_id = paper['y'].tabid

        if args.handle_error and not any([d["tab_id"] == tab_id for d in error_data]):
            continue

        # Use gold_caption if the difficulty is medium
        gold_caption = None
        if args.difficulty == "medium":
            gold_caption = paper['y'].caption
        
        # Save path for all experiments in this setting
        save_path = f"{args.results_path}/{tab_id}/{args.difficulty}/{args.endtoend.model_type}/{args.endtoend.name}/{args.retry_type}"

        # For each try/experiment, run the endtoend pipeline
        for r_id in range(retry_num):
            if args.handle_error:
                # find the index of the error in error_data
                error_index = [i for i, d in enumerate(error_data) if d["tab_id"] == tab_id and d["r_id"] == r_id]
                if len(error_index) == 0:
                    continue
                error_index = error_index[0]

            indices = list(range(len(paper['x'])))
            if args.retry_type == "shuffle":
                random.Random(r_id).shuffle(indices)
            paper['x'] = [paper['x'][i] for i in indices]
            
            # TODO: Set temperature for each retry
            if args.retry_type == "temperature":
                pass

            column_num = len(paper['y'].schema)

            table_set = endtoend(args=args, sample=paper, tab_id=tab_id, index=index, column_num=column_num, gold_caption=gold_caption)
            print("\nfinal_output", table_set)

            is_error = False
            for table in table_set:
                if "text" in table:
                    is_error = True
                    break
            
            if not is_error:
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)            
                with open(f"{save_path}/try_{r_id}.json", "w") as f:
                    json.dump(table_set, f)
                print(f"Results saved to: {save_path}/try_{r_id}.json\n")

                # Remove the error from the error file
                if args.handle_error:
                    error_data.pop(error_index)
                    with open(error_save_path, "w") as f:
                        json.dump(error_data, f)
            else:
                print(f"Error in paper_idx_{index} try_{r_id}\n")
                if args.handle_error:
                    continue
                error_data.append({
                    "tab_id": tab_id,
                    "r_id": r_id
                })
                with open(error_save_path, "w") as f:
                    json.dump(error_data, f)


if __name__ == "__main__":
    main()