"""Entrypoint for running experiments."""

from datetime import datetime
import os
import sys
import random
import json
import traceback
from typing import Any, Dict, List
from concurrent.futures import ProcessPoolExecutor

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

def process_paper(
    args: DictConfig,
    index: int,
    paper: Dict[str, Any],
    endtoend: Any,
    retry_num: int,
    error_path: str,
    error_data: List[Dict[str, Any]]
):
    tab_id = paper['y'].tabid
    print(f"[START] tab_id: {tab_id}")
    try:
        # If processing errors, skip if this tab_id is not in error_data
        if args.handle_error and not any([d["tab_id"] == tab_id for d in error_data]):
            return True, tab_id, None

        # Use gold_caption if the difficulty is medium
        gold_caption = None
        if args.difficulty == "medium":
            gold_caption = paper['y'].caption
        
        # Save path for all experiments in this setting
        save_path = f"{args.results_path}/{tab_id}/{args.difficulty}/{args.endtoend.model_type}/{args.endtoend.name}/{args.retry_type}"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # For each try/experiment, run the endtoend pipeline
        for r_id in range(retry_num):
            start_time = datetime.now()

            # If processing errors, get index of the error in error_data
            if args.handle_error:
                error_index = [i for i, d in enumerate(error_data) if d["tab_id"] == tab_id and d["r_id"] == r_id]
                if len(error_index) == 0:
                    continue
                error_index = error_index[0]
            # If this tab_id, r_id pair is already processed, skip
            elif os.path.exists(f"{save_path}/try_{r_id}.json"):
                print(f"\tretry_id: {r_id} already exists")
                continue

            indices = list(range(len(paper['x'])))
            if args.retry_type == "shuffle":
                random.Random(r_id).shuffle(indices)
            paper['x'] = [paper['x'][i] for i in indices]
            
            # TODO: Set temperature for each retry
            if args.retry_type == "temperature":
                pass

            column_num = len(paper['y'].schema)

            table_set = endtoend(args=args, sample=paper, tab_id=tab_id, index=index, column_num=column_num, gold_caption=gold_caption)

            # If there is an error in the output, save the error file and continue
            is_error = False
            for table in table_set:
                if "text" in table:
                    is_error = True
                    break
            
            if not is_error:   
                # Remove error file if it was processed successfully
                if args.handle_error:
                    os.remove(f"{error_path}/{tab_id}___{r_id}.json")
                    prev_error_counts = error_data[error_index]["error_counts"]
                    for key, value in prev_error_counts.items():
                        if key in table_set["error_counts"]:
                            table_set["error_counts"][key] += value
                        else:
                            table_set["error_counts"][key] = value
                with open(f"{save_path}/try_{r_id}.json", "w") as f:
                    json.dump(table_set, f)
                print(f"\tResults saved to: {save_path}/try_{r_id}.json")
            else:
                print(f"\tError in paper_idx_{index} try_{r_id}")
                # Save error file
                if args.handle_error:
                    prev_error_counts = error_data[error_index]["error_counts"] if args.handle_error else {}
                    for key, value in prev_error_counts.items():
                        if key in table_set["error_counts"]:
                            table_set["error_counts"][key] += value
                        else:
                            table_set["error_counts"][key] = value
                with open(f"{error_path}/{tab_id}___{r_id}.json", "w") as f:
                    json.dump({ "tab_id": tab_id, "r_id": r_id, "error_counts": table_set["error_counts"] }, f)

            print(f"\tretry_id: {r_id} / time taken (s): {(datetime.now() - start_time).total_seconds()}")
            break
        print(f"[COMPLETE] tab_id: {tab_id}")
        return True, tab_id, None
    except Exception as e:
        print(f"[ERROR] tab_id: {tab_id} ({e})")
        return False, tab_id, traceback.format_exc()


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
    retry_num = 1
    # tables = load_outputs(args)

    error_path = f"{args.results_path}/errors/{args.difficulty}/{args.endtoend.model_type}/{args.endtoend.name}/{args.retry_type}"
    if not os.path.exists(error_path):
        os.makedirs(error_path, exist_ok=True)
    # get all error files in the error_path
    error_files = [f for f in os.listdir(error_path) if os.path.isfile(os.path.join(error_path, f))]
    error_data = []
    for error_file in error_files:
        with open(f"{error_path}/{error_file}", "r") as f:
            error_data += json.load(f)

    # Process batch_size papers in parallel
    batch_size = 10
    futures = []
    with ProcessPoolExecutor(max_workers=batch_size) as executor:
        for index, paper in enumerate(data[:100]):
            future = executor.submit(
                process_paper, 
                args,
                index,
                paper,
                endtoend,
                retry_num,
                error_path,
                error_data
            )
            futures.append(future)
    
    errors = []
    for future in futures:
        result, tab_id, error = future.result()
        # Save non-code related errors
        if not result:
            # get full error message (file name, line number, etc.)
            errors.append({ "tab_id": tab_id, "error": error })
    if len(errors) > 0:
        print(f"OVERALL ERRORS")
        print(json.dumps(errors, indent=4))

if __name__ == "__main__":
    main()