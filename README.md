# Paper Comparison

## Installation
```
pip install -e ".[dev]"
```

## Running

When running, you need to pass some input example, and either a baseline or a schematizer and populator.

To run an end-to-end model on a single example:
```
python paper_comparison/run.py data.path=data/debug_abstracts.jsonl endtoend=debug
```

To run a schematizer-populator combination on a single example:
```
python paper_comparison/run.py data.path=data/debug_abstracts.jsonl schematizer=followup_questions populator=qa
```

After running, the results path will be output to the terminal. This path contains three files:
 - `tables.jsonl`: a file with one table per line
 - `metrics.jsonl`: a file with table metrics
 - `commands.txt`: a list of commands that were run to obtain the results in the directory. Most recent at the bottom.

## Training
(Not implemented yet)
For training, train one of `endtoend`, `schematizer`, or `populator`. The relevant configs must have training details in the `training` field of the config.

Training an end-to-end model:
```
python paper_comparison/train.py train_data=<data_config> val_data=<data_config> endtoend=<endtoend_config> trainer=<train_config>
```

Training a schematizer model:
```
python paper_comparison/train.py train_data=<data_config> val_data=<data_config> schematizer=<schematizer_config>
```

Training a populator model requires specifying a schematizer to use as well:
```
python paper_comparison/train.py train_data=<data_config> val_data=<data_config> schematizer=<schematizer_config> populator=<populator>
```

## Code Structure
- `loaders`: Code for loading in tables (in json format) and papers (using papermage)
- `endtoend`: Transform collections of papers directly into tables.
- `schematizers`: Transform collections of papers into schema
- `populators`: Tansforms papers + schema into table
- `eval`: Evaluates tables