from collections import defaultdict

from argparse import ArgumentParser
import json

import pandas as pd


def main():
    argp = ArgumentParser()
    argp.add_argument("in_tables_path", type=str)
    argp.add_argument("out_tables_path", type=str)
    argp.add_argument("out_papers_path", type=str)
    argp.add_argument("--bib_entries_path", type=str, default="../arxiv_dump/out_bib_entries.jsonl")
    args = argp.parse_args()

    with open(args.in_tables_path) as f:
        in_tables = [json.loads(line.strip()) for line in f]

    with open(args.bib_entries_path) as f:
        bib_entries_list = [json.loads(line.strip()) for line in f]
        bib_entries = {entry["corpus_id"]: entry for entry in bib_entries_list}
        bib_entries_by_bib_hash = {entry["bib_hash_or_arxiv_id"]: entry for entry in bib_entries_list}

    # tables.jsonl
    bib_table_map = defaultdict(list)

    with open(args.out_tables_path, "w") as f:
        for table in in_tables:
            table_json = {}
            table_json["tabid"] = table["_table_hash"]
            if not table["table_json"]["table_dict"]:
                print(f"Skipping {table['_table_hash']}")
                continue

            df = pd.DataFrame(table["table_json"]["table_dict"])
            # rewrite the "References" to have the *corpus_ids* rather than the *bib_hash* prefixes or *arxiv_ids*
            row_subset = []
            for row in table["row_bib_map"]:
                if row["corpus_id"] != -1:
                    df["References"][row["row"]] = row["corpus_id"]
                    row_subset.append(row["row"])
            df = df.iloc[row_subset].set_index("References")

            df = df.map(
                lambda x: x if isinstance(x, list) else [x]
            )  # this is a silly formatting thing - table values are lists. not sure if we'll keep it or not
            table_json["table"] = df.to_dict()
            table_json["row_bib_map"] = table["row_bib_map"]
            for row_entry in table["row_bib_map"]:
                bib_table_map[row_entry["corpus_id"]].append(table["_table_hash"])
            f.write(json.dumps(table_json) + "\n")

    # papers.jsonl

    print()
    # breakpoint()
    with open(args.out_papers_path, "w") as f:
        num_skipped = 0
        for corpus_id in bib_table_map:
            paper = {}
            if corpus_id not in bib_entries:
                continue
            try:
                paper["tabids"] = bib_table_map[corpus_id]
                # paper["bib_hash_or_arxiv_id"] = bib_entries[corpus_id]["bib_hash_or_arxiv_id"]
                paper["corpus_id"] = corpus_id  # bib_entries[corpus_id]["corpus_id"]
                if "metadata" in bib_entries[corpus_id]:
                    paper["title"] = bib_entries[corpus_id]["metadata"].get("title", bib_entries.get("title"))
                    paper["paper_id"] = bib_entries[corpus_id]["metadata"].get("paperId")
                else:
                    paper["title"] = None
                    paper["paper_id"] = -1
            except KeyError:
                print(bib_entries[corpus_id].keys())
            except AttributeError:
                # print(f"Skipping {bib_hash}")
                num_skipped += 1
                continue
            # else:
            #     num_hits += 1
            #     print(":checkmark:")
            f.write(json.dumps(paper) + "\n")


if __name__ == "__main__":
    main()
