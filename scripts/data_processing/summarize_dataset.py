from argparse import ArgumentParser
from collections import Counter
import json

import numpy as np


def print_summary(datas, names, sep=" "):
    # form the table
    headers = ["", "Min", "Max", "Median", "Mean", "Total"]
    out_table = np.empty((len(names) + 1, len(headers)), dtype=object)
    for i in range(len(headers)):
        out_table[0][i] = headers[i]
    for data_i, data in enumerate(datas):
        out_table[1 + data_i][0] = names[data_i]
        out_table[1 + data_i][1] = str(np.min(data))
        out_table[1 + data_i][2] = str(np.max(data))
        out_table[1 + data_i][3] = str(np.median(data))
        out_table[1 + data_i][4] = f"{np.mean(data):.3f}"
        out_table[1 + data_i][5] = str(sum(data))

    # next, format out_table by column
    for i in range(len(headers)):
        max_col_width = max([len(cell) for cell in out_table[:, i]])
        padding_str = "{" + f": >{max_col_width}" + "}"
        for j in range(len(names) + 1):
            out_table[j, i] = padding_str.format(out_table[j, i])

    for row in out_table:
        print(*row, sep=sep)
    return
    # print(out_table)
    # return
    for data, name in zip(datas, names):
        pass

    max_name_len = max([len(name) for name in names])
    padding_str = "{" + f": >{max_name_len}" + "}"
    headers = [padding_str.format(""), "Min", "Max", "Median", "Mean", "Total"]
    # max_header_size = max([len(header) for header in headers])
    header_padding_strs = ["{" + f": >{len(header)}" + "}" for header in headers[1:]]
    print(*headers, sep=sep)
    for data, name in zip(datas, names):
        print(
            padding_str.format(name),
            header_padding_strs[0].format(np.min(data)),
            header_padding_strs[1].format(np.max(data)),
            header_padding_strs[2].format(np.median(data)),
            header_padding_strs[3].format(f"{np.mean(data):.3f}"),
            header_padding_strs[4].format(sum(data)),
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
    return value in {"-", "–", "-", "n/a"}


def get_aspect_type(column):
    if "{{cite:" in column[0]:
        return "cite"

    if all([is_numeric(val) or is_na(val) for val in column]):
        return "num"

    if all([is_binary(val) or is_na(val) for val in column]):
        return "bool"

    if len(set(column)) == 1:
        # meant for grouping
        return "cat-uniq"

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

    # 0. Print size of dataset
    print(f"Number of instances: {len(dataset)}\n")
    # 1. Compute number of rows, columns, and unique rows for the dataset
    separator = " & " if args.latex else "  "
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

    # 2. Compute how many unique papers are considered by the tables using corpus ids
    unique_corpus_ids = set()
    missing_corpus_ids = 0
    for table in dataset:
        for row in table["row_bib_map"]:
            if row["corpus_id"] == -1:
                missing_corpus_ids += 1
                continue
            unique_corpus_ids.add(row["corpus_id"])
    print(f"Number of unique corpus ids among all tables: {len(unique_corpus_ids)}")
    print(f"Number of papers missing corpus ids: {missing_corpus_ids}")
    print()

    # 3. Compute the distribution of aspect types
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

    print("\nNumber of n/a's:")
    num_nas_total = 0
    num_values_total = 0
    for table in dataset:
        num_values_total += sum([len(val) for val in table["table_json"]["table_dict"].values()])
        num_nas_total += sum([len([v for v in val if is_na(v)]) for val in table["table_json"]["table_dict"].values()])
    print(f"{num_nas_total} / {num_values_total} ({num_nas_total/num_values_total:.3%})")


if __name__ == "__main__":
    main()
