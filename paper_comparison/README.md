# Table Generation Pipeline

## Baseline

Navigate to the `configs/config.yaml`. Here, you can choose between `endtoend` options by selecting either `ours_outputs` or `baseline_outputs`.

If you wish to change the model type or the number of tables you wna to generate per try, you can do so by modifying the `model_type` and `num_commonality` parameter in either the `configs/endtoend/baseline_outputs.yaml` or `configs/endtoend/ours_outputs.yaml`. (Other hyperparameters are fixed in current experiment setup.)

To run a baseline model on a single example:
```
python paper_comparison/run.py
```
After running, the results path will be output to the terminal. This path contains three files:
 - `tables.jsonl`: a file with one table per line (All predefined baseline types are included)
 - `commands.txt`: a list of commands that were run to obtain the results in the directory. Most recent at the bottom.
