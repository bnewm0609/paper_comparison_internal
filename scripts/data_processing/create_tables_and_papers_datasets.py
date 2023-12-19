from collections import defaultdict

from argparse import ArgumentParser
import json

import pandas as pd


def main():
    argp = ArgumentParser()
    argp.add_argument("in_tables_path", type=str)
    argp.add_argument("out_tables_path", type=str)
    argp.add_argument("out_papers_path", type=str)
    args = argp.parse_args()

    with open(args.in_tables_path) as f:
        in_tables = [json.loads(line.strip()) for line in f]

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
            df = df.set_index(df.columns[0])

            # if the citations are in the columns, then change them to the rows
            if " ".join(df.columns).count("{{cite:") > 0:
                df = df.transpose()

            df = df.map(
                lambda x: [x]
            )  # this is a silly formatting thing - table values are lists. not sure if we'll keep it or not
            table_json["table"] = df.to_dict()
            table_json["bib_hashes"] = table["bib_hash"]
            for bib_hash in table["bib_hash"]:
                bib_table_map[bib_hash].append(table["_table_hash"])
            f.write(json.dumps(table_json) + "\n")

    # papers.jsonl
    with open("arxiv_dump/out_bib_entries.jsonl") as f:
        bib_entries = [json.loads(line.strip()) for line in f]
        bib_entries = {entry["bib_hash"]: entry for entry in bib_entries}

    print()
    with open(args.out_papers_path, "w") as f:
        num_skipped = 0
        for bib_hash in bib_table_map:
            paper = {}
            if bib_hash not in bib_entries:
                continue
            try:
                paper["tabids"] = bib_table_map[bib_hash]
                paper["corpus_id"] = bib_entries[bib_hash]["corpus_id"]
                if "metadata" in bib_entries[bib_hash]:
                    paper["title"] = bib_entries[bib_hash]["metadata"].get("title", bib_entries.get("title"))
                    paper["paper_id"] = bib_entries[bib_hash]["metadata"].get("paperId")
                else:
                    paper["title"] = None
                    paper["paper_id"] = -1
            except KeyError:
                print(bib_entries[bib_hash].keys())
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
