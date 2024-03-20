from argparse import ArgumentParser
from collections import Counter
import json

import numpy as np


def print_summary(datas, names, sep="\t"):
    max_name_len = max([len(name) for name in names])
    padding_str = "{" + f": >{max_name_len}" + "}"
    print(padding_str.format(""), "Min", "Max", "Median", "Mean", sep=sep)
    for data, name in zip(datas, names):
        print(
            padding_str.format(name),
            np.min(data),
            np.max(data),
            np.median(data),
            f"{np.mean(data):.3f}",
            sep=sep,
        )


import re


def is_numeric(value):
    """
    Filters out values that contain some common numeric values that don't need context.
    They include units like 'K' for thousand, 'M' for million and others.
    These are *not* comprehensive, and are curated only on the tables evaluated
    so far from the high quality 2308 papers.
    """
    value = value.replace("below", "<")
    value = re.sub(r"[<$∼~∼\s]", "", value.lower().strip())
    units_regex = r"[<-]?\d+[km]"
    time_regex = r"(\d+(?:hrs?|hours?))?(\d+(?:ms?|mins?|minutes?))?(\d+(?:s|sec|seconds?))?"
    freq_regex = r"\d+[gm]hz"
    if (
        re.match(units_regex, value) is not None
        or any(re.match(time_regex, value).groups())
        or re.match(freq_regex, value) is not None
    ):
        return True
    # this is dumb, but easy and fast
    try:
        float(value.replace(",", "").replace("-", ""))
        return True
    except ValueError:
        return False


def is_binary(value):
    value = value.strip()
    return value.lower() in {"yes", "no"} or value in {"✘", "✗", "×", "✔", "✓"}


def is_na(value):
    value = re.sub("[∼~\s]", "", value.lower())
    return value in {"-", "–", "-"}


def get_aspect_type(column):
    if "{{cite:" in column[0]:
        return "cite"

    if all([is_numeric(val) or is_na(val) for val in column]):
        return "num"

    if all([is_binary(val) or is_na(val) for val in column]):
        return "bool"

    if len(set(column)) != len(column):
        # meant for grouping
        return "cat"

    # use length as a heuristic to determine if its gen or ent
    val_lens = [len(val.split()) > 4 for val in column if not is_na(val)]
    if sum(val_lens) > len(val_lens) / 2:
        return "gen"
    else:
        return "ent"


def main():
    argp = ArgumentParser()
    argp.add_argument("dataset_path", type=str)
    argp.add_argument("--latex", action="store_true")
    args = argp.parse_args()

    with open(args.dataset_path) as f:
        dataset = [json.loads(line) for line in f]

    # 1. compute number of rows, columns, and unique rows for the dataset
    separator = " & " if args.latex else "\t"
    print_summary(
        [
            [len(tab["table_json"]["table_dict"]["References"]) for tab in dataset],
            [len(set(tab["table_json"]["table_dict"]["References"])) for tab in dataset],
            [len(tab["table_json"]["table_dict"]) for tab in dataset],
        ],
        ["Papers", "Papers (uniq)", "Aspects"],
        sep=separator,
    )
    print()

    # 2. Compute the distribution of aspect types
    table_aspect_labels = {}
    all_aspect_labels = []
    for table_i, table in enumerate(dataset):
        tab_dict = table["table_json"]["table_dict"]
        aspect_labels = []
        for col in tab_dict:
            aspect_labels.append(get_aspect_type(tab_dict[col]))
        table_aspect_labels[table["_table_hash"]] = aspect_labels
        all_aspect_labels.extend(aspect_labels)

    print("Distribution of types of aspects:")
    label_dist = Counter(all_aspect_labels)
    del label_dist["cite"]
    label_dist = label_dist.most_common()
    total_labels = sum([count for _, count in label_dist])
    for label, count in label_dist:
        print(label, count, f"{count/total_labels:.1%}", sep=separator)


if __name__ == "__main__":
    main()
