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

    
    for index, paper in enumerate(data):
        tab_id = paper['y'].tabid

        # Load the experiment settings if they exist
        experiment_settings_path = f"{args.results_path}/{tab_id}/experiment_settings_{args.retry_type}.json"
        experiment_settings = None
        if not os.path.exists(experiment_settings_path):
            os.makedirs(f"{args.results_path}/{tab_id}", exist_ok=True)  
            
        if os.path.exists(experiment_settings_path):
            with open(experiment_settings_path, "r") as f:
                experiment_settings = json.load(f)
                
        else:
            # Create the experiment settings if they don't exist
            experiment_settings = []
            len_papers = len(paper['x'])
            for retry_id in range(retry_num):
                # shuffle paper index and save the shuffled index
                indexed_list = list(range(len_papers))
                random.shuffle(indexed_list)
                experiment_settings.append({
                    "id": retry_id,
                    "indices": indexed_list if args.retry_type == "shuffle" else list(range(len_papers)),
                    "temperature": 1.0 if args.retry_type == "shuffle" else 0.7
                    # TODO: Change default temperature and shuffle temperature depending on experiment
                })
            with open(experiment_settings_path, "w") as f:
                json.dump(experiment_settings, f)
                
        # Use gold_caption if the difficulty is medium
        gold_caption = None
        if args.difficulty == "medium":
            gold_caption = paper['y'].caption
        
        # Save path for all experiments in this setting
        save_path = f"{args.results_path}/{tab_id}/{args.difficulty}/{args.endtoend.name}/{args.retry_type}"

        # For each try/experiment, run the endtoend pipeline
        for retry_id, setting in enumerate(experiment_settings):
            paper['x'] = [paper['x'][i] for i in setting['indices']]
            column_num = len(paper['y'].schema)
            # TODO: Set temperature for each retry
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
                with open(f"{save_path}/try_{retry_id}.json", "w") as f:
                    json.dump(table_set, f)
                print(f"Results saved to: {save_path}/try_{retry_id}.json\n")
            else:
                pass
                # TODO: handle error case
                # handle_error_cases(args, table_set, tab_id, index, retry_id, save_path) 
                
    
if __name__ == "__main__":
    main()