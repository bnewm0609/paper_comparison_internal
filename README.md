# Paper Comparison

## Installation
```
pip install -e ".[dev]"
```

## Running

When running, you need to pass some input example, and either a baseline or a schematizer and populator.

To run an end-to-end model on a single example:
```
python paper_comparison/run.py data.path=data/debug_abstracts endtoend=debug
```

To run a schematizer-populator combination on a single example:
```
python paper_comparison/run.py data.path=data/debug_abstracts schematizer=followup_questions populator=qa
```

After running, the results path will be output to the terminal. This path contains three files:
 - `tables.jsonl`: a file with one table per line
 - `metrics.jsonl`: a file with table metrics
 - `commands.txt`: a list of commands that were run to obtain the results in the directory. Most recent at the bottom.

## Data
Currently all data is in sub-directories of the `data` path. These directories are passed to the `data.path` field at the command line. In these directories, there are one or two files: one (usually called `papers.jsonl`) has one paper json object per line. For example, one line will have the following fields:
```json
{
    "tabids": ["<Table IDs. All papers that will be compared should have the same Table ID>. This is a list because papers can"],
    "paperid": "<Paper ID. A string used as the name of the row in the table. It's used to link tables to papers."
    <other information eg abstracts, title, authors, etc.>
}
```

If this data is being used for training or supervised evaluation, there will also be gold tables in this directory (in a file usually called `tables.jsonl`). Each line in the table has the following structure:
```json
{
    "tabid" : "<Table ID",
    "table": {
        "row1": {"paperid_1": "value", "paperid_2": "value"},
        "row2": {"paperid_1": "value", "paperid_2": "value"},
        ...
    }
}
```

Each of these files can be overridden individually by setting the `data.papers_path` or `data.tables_path` at the command line. E.g.:
```
python paper_comparison/run.py data.papers_path=data/debug_abstracts/papers.jsonl data.tables_path=data/debug_abstracts/tables.jsonl endtoend=debug
```

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
- `loaders`: Code for loading in tables (in json format) and papers (using papermage) [Not implemented yet]
- `endtoend.py`: Transform collections of papers directly into tables. Used for baselines.
- `schematizer.py`: Transform collections of papers into schema
- `populator.py`: Tansforms papers + schema into table
- `eval.py`: Evaluates tables
- `utils.py`: Various experiment utilities.
