# data

### gold data

We have 3 sets of gold data:
```
|-- highest_quality_1k
|-- metric_validation_0     # << nearly all the experiments on this
|-- metric_validation_1
```

Opening up `metric_validation_0/`, we can see it's also made of multiple files:
```
|-- dataset_with_ics.jsonl
|-- papers.jsonl
|-- tables.jsonl
```

Opening up `tables.jsonl`

```python
import json

with open('/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/tables.jsonl', 'r') as f:
    tables = [json.loads(line) for line in f]
```

#### Q: How many tables?
```python
len(tables)
>> 557
```

#### Q: What is shape of each gold table?
```
{
    "tabid": <str>,
    "table": {
        "COLUMN_NAME_1": {"CORPUS_ID": ["VALUE"]},      # len==1
        "COLUMN_NAME_2": {"CORPUS_ID": ["VALUE"]},
        "COLUMN_NAME_3": {"CORPUS_ID": ["VALUE"]},
    },
    "row_bib_map": [
        {...},          # contains title, abstract, corpus_id, row_id, bib_hash
        {...},
        {...}
    ]
}
```

Weird, there's no caption or inline reference data. I'll check `dataset_with_ics.jsonl` for that.

#### Q: Where to get caption or inline reference data?

```python
import json

with open('/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/dataset_with_ics.jsonl', 'r') as f:
    tables = [json.loads(line) for line in f]

# still same length
assert len(tables) == 557
```

#### Q: What's shape of each gold table (with ICS data)?

Wow so much different...

```
{
    "paper_id": <str>,        # points to paper containing this table (arXiv id)
    "_table_hash": <str>,     # i think this maps to `tabId` field in `tables.jsonl`
    "caption": <str>,
    "in_text_ref": [
        {..."text": <str>, ...},    # like a paragraph of text
        {..."text": <str>, ...},
        {..."text": <str>, ...},
    ],
    "table_json": {...}       # i think this is everything u can get above
}
```

Let's double-check that you can get everything about `tables.jsonl` from `dataset_with_ics.jsonl`

```python
import json

with open('/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/tables.jsonl', 'r') as f:
    tables = [json.loads(line) for line in f]

table_id_to_info = {
    table['tabid']: {
        "columns": sorted(table['table'].keys()),
        "papers": [bib["corpus_id"] for bib in table['row_bib_map']]
    } for table in tables
}


with open('/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/dataset_with_ics.jsonl', 'r') as f:
    dataset = [json.loads(line) for line in f]

for table in dataset:
    # check ids match exactly
    assert table["_table_hash"] in table_id_to_info

    # check input papers match exactly
    assert len(table["row_bib_map"]) == len(table_id_to_info[table["_table_hash"]]["papers"])
    for paper in table["row_bib_map"]:
        assert paper["corpus_id"] in table_id_to_info[table["_table_hash"]]["papers"]

    # check columns match exactly
    # turns out in the `dataset` version, there is an extra column called "References"
    assert "References" in table["table_json"]["table_dict"].keys()

    # if you remove this, do the other columns match exactly?
    all_columns = [colname for colname in table["table_json"]["table_dict"].keys() if colname != "References"]
    assert len(all_columns) == len(table_id_to_info[table["_table_hash"]]["columns"])
    for column in all_columns:
        assert column in table_id_to_info[table["_table_hash"]]["columns"]
```

Ok, so basically we only need to develop off the `dataset_with_ics.jsonl` file.



### pred data

```python
import json


def get_instance_dicts(pred_dir: str) -> list[dict]:
    """Just a function that traverses paths in a directory"""
    instance_dicts = []
    for instance_id in os.listdir(pred_dir):
        dir0 = os.path.join(pred_dir, instance_id)
        # skips random files
        if not os.path.isdir(dir0):
            continue
        for base_model in os.listdir(dir0):
            # skips random files
            dir1 = os.path.join(dir0, base_model)
            if not os.path.isdir(dir1):
                continue
            for prompt_method in os.listdir(dir1):
                dir2 = os.path.join(dir1, prompt_method)
                # one more time, random files
                if not os.path.isdir(dir2):
                    continue
                for try_num in os.listdir(dir2):
                    instance_dicts.append(
                        {
                            "instance_id": instance_id,
                            "base_model": base_model,
                            "prompt_method": prompt_method,
                            "try_num": try_num,
                            "path": os.path.join(dir2, try_num),
                        }
                    )
    return instance_dicts


instance_dicts = get_instance_dicts(pred_dir="/Users/kylel/ai2/paper_comparison_internal/final_data/value_preds/metric_validation_0_setting5/")

for model in ["gpt3.5", "mixtral"]:
    for prompt_method in ["ours_outputs"]:
        for try_num in ["try_0_with_values.json"]:
            n = len([i for i in instance_dicts if i["base_model"] == model and i["prompt_method"] == prompt_method and i["try_num"] == try_num])
            print(f"{model} {prompt_method} {try_num} {n}")
```

Ok weird, looks like each of these predictions is smaller than the gold data:

```
gpt3.5 ours_outputs try_0_with_values.json 555
mixtral ours_outputs try_0_with_values.json 553
```

What's missing instances?
```python
# first, filter to only predictions that actually completed properly
instance_dicts = [i for i in instance_dicts if i["try_num"] == "try_0_with_values.json"]

gpt3_5_instance_dicts = [i for i in instance_dicts if i["base_model"] == "gpt3.5"]
mixtral_instance_dicts = [i for i in instance_dicts if i["base_model"] == "mixtral"]

# check if there are any predictions that don't have a gold
for i in gpt3_5_instance_dicts:
    if i["instance_id"] not in table_id_to_info:
        raise Exception(f"Instance {i['instance_id']} not in gold data")
for i in mixtral_instance_dicts:
    if i["instance_id"] not in table_id_to_info:
        raise Exception(f"Instance {i['instance_id']} not in gold data")
print(f"All predictions have a gold")

# now, check if there are any gold tables that don't have a prediction
for table_id in table_id_to_info.keys():
    if table_id not in {i["instance_id"] for i in gpt3_5_instance_dicts}:
        print(f"Table {table_id} not in predictions for gpt3.5")
    if table_id not in {i["instance_id"] for i in mixtral_instance_dicts}:
        print(f"Table {table_id} not in predictions for mixtral")
```


