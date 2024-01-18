# Paper Comparison

## Installation
```
pip install -e ".[dev]"
```

## High quality initial dataset

For the data in `data/arxiv_tables_2308_high_quality`, there are three relevant files:
- `tables.jsonl`: a list of 30 papers. Each line is a `json` object with the following fields:
    - `tabid`: a unique hash for the table that comes from the latex-processing pipeline
    - `table`: a python dictionary dump of the table created from`pd.DataFrame.to_dict()`. Can be loaded back into pandas with `pd.DataFrame(table['table'])`. The index of the data frame is the the `corpus_id` of the full text of the paper (which can be accessed at `data/full_texts/{corpus_id}.jsonl`).
    - `row_bib_map`: a list of `json` objects, where each represents a citation from the table. (This shouldn't be needed anymore, but I'm leaving it in in case it's useful for now. Will probably be deleted soon.) Each object contains:
        - the `corpus_id` of the full text of the paper
        - the `row` in the table that `corpus_id` corresponds to
        - the `type` of the citation - `"ref"` means it's an external reference and `"ours"` means that the paper containing the table is represented in the given row.
        - the `bib_hash_or_arxiv_id` associated with that corpus_id. `arxiv_id` is used when the `type` is `"ours"` and the `bib_hash` is used when it's `"ref"`. The `bib_hash` comes from the latex-processing pipeline and it's a function of the citation text and the citing paper's arxiv_id.
- `papers.jsonl`: a list of all the papers needed for generating the tables. Each paper entry contains:
    - `tabids`: a list of ids for the tables the paper is cited in.
    - `corpus_id`, the corpus id of a paper. If the full text is available, it can be accessed at `data/full_texts/{corpus_id}.jsonl`. (For the small subset, all of these texts should be available.)
    - `title`: the title of the paper from s2.
    - `paper_id`: the s2 id of the paper.
- `dataset.jsonl`: contains a superset of the information in `tables.jsonl` (it additionally includes an html version of the table from which the json format was derived, some artefacts from the html->json conversion, and some additional metadata about the papers and tables)

These are loaded in using the `configs/full_texts` config, with the `data.FullTexts` class performing the loading part.

```
python paper_comparison/run.py data=full_texts data.path=data/arxiv_tables_2308_high_quality endtoend=debug endtoend.name=oracle
```
(Setting `endtoend.name=oracle` just has the system output the gold tables.)

## ArXiv data processing
See `scripts/data_processing`

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
