# Table Generation Pipeline

## Baseline

Navigate to the `configs/config.yaml`. Here, you can choose between `endtoend` options by selecting either `ours_outputs` or `baseline_outputs`.

If you wish to change the model type or the number of tables you want to generate per try, you can do so by modifying the `model_type` and `num_commonality` parameter in either the `configs/endtoend/baseline_outputs.yaml` or `configs/endtoend/ours_outputs.yaml`. (Other hyperparameters are fixed in current experiment setup.)

If you want to change the input data, you can do so by modifying the `path` parameter in the `configs/data/abstracts.yaml`.

To run a baseline model on a single example:
```
python paper_comparison/run.py
```
After running, the results of each try and the path will be output to the terminal. This path contains three files:
 - `results/{args.data._id}/{args.endtoend.difficulty}/{args.endtoend.model_type}/{args.mode}/{args.endtoend.retry_type}`: ten files `try_{idx}.json` with `num_commonality` number of tables.
 - `results/{args.data._id}`: a file with experiment setting that used for generating the table.
 - `commands.txt`: a list of commands that were run to obtain the results in the directory. Most recent at the bottom.
