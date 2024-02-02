# Table Generation Pipeline

## Baseline

To run the baseline, first change the `endtoend` parameter in `configs/config.yaml` to `baseline_outputs`.

Then, edit the `baseline_type` parameter in `configs/endtoend/baseline_outputs.yaml` to choose the types of baseline prompts that you want to run.

To run a baseline model on a single example:
```
python paper_comparison/run.py
```
After running, the results path will be output to the terminal. This path contains three files:
 - `tables.jsonl`: a file with one table per line (All predefined baseline types are included)
 - `commands.txt`: a list of commands that were run to obtain the results in the directory. Most recent at the bottom.
