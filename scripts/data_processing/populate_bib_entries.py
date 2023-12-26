from argparse import ArgumentParser
import json
from pathlib import Path
import re
import requests_cache
import requests
import time

from tqdm import trange

BATCH_SIZE = 100


session = requests_cache.CachedSession("titles_ids_cache")


def get_titles_s2_internal(citations_batch):
    """
    Must be run while on S2 network
    """
    base_url = "http://bibentry-predictor.v0.dev.models.s2.allenai.org/invocations"
    data_json = {
        "instances": [{"bib_entry": citation_text} for citation_text in citations_batch],
        "metadata": {},
    }
    response = session.post(base_url, json=data_json)
    predictions = response.json()
    try:
        titles = [pred["title"] for pred in predictions["predictions"]]
    except KeyError:
        print("No key 'predictions' or 'title' found in response!")
        print(predictions)
        raise (KeyError)
    return titles


def get_corpus_ids_s2_internal(titles_batch):
    """
    Must be run while on S2 network
    """
    base_url = "http://pipeline-api.prod.s2.allenai.org/citation/match"
    data_json = [{"title": title if title is not None else "None"} for title in titles_batch]
    response = session.post(base_url, json=data_json)
    try:
        corpus_ids = response.json()
    except requests.exceptions.JSONDecodeError:
        breakpoint()
    return corpus_ids


def get_metadata_s2_public(corpus_ids_batch, prefix="CorpusId:"):
    """
    Can be run off of S2 network
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=corpusId,isOpenAccess,openAccessPdf,externalIds,title,abstract"
    ids_json = {"ids": [f"{prefix}{corpus_id}" for corpus_id in corpus_ids_batch]}
    response = session.post(base_url, json=ids_json)
    metadata = response.json()
    return metadata


def main():
    argp = ArgumentParser()
    argp.add_argument(
        "papers_path",
        type=str,
        help="something in the out_xml directory. Has a map between bib hashes and citations",
    )
    argp.add_argument(
        "dataset_path",
        type=str,
        help="something in the out_xml_filtered directory. A dataset file that associates each table with some bib hashes",
    )
    argp.add_argument("out_path", type=str)
    args = argp.parse_args()

    all_bib_entries = {}
    with open(args.papers_path) as f:
        for line in f:
            paper = json.loads(line)
            # Extracts the bib_entities
            all_bib_entries |= paper["bib_entries"]

    # Subsets the ones that we need based on the dataset
    all_bib_hashes = set()
    all_arxiv_ids = set()
    with open(args.dataset_path) as f:
        for line in f:
            sample = json.loads(line)
            all_bib_hashes.update([bib_hash for bib_hash in sample["bib_hash"] if bib_hash in all_bib_entries])
            # also add the arxiv id for the paper
            all_arxiv_ids.add(sample["paper_id"])

    # Filters out the bib_hashes/arxiv_ids we've already saved
    if Path(args.out_path).exists():
        with open(args.out_path) as f:
            prev_bib_hash_dict = [json.loads(line) for line in f]
            prev_bib_hashes = set(entry["bib_hash_or_arxiv_id"] for entry in prev_bib_hash_dict)
        all_bib_hashes = [bib_hash for bib_hash in all_bib_hashes if bib_hash not in prev_bib_hashes]
        all_arxiv_ids = [arxiv_id for arxiv_id in all_arxiv_ids if arxiv_id not in prev_bib_hashes]
    else:
        all_bib_hashes = list(all_bib_hashes)
        all_arxiv_ids = list(all_arxiv_ids)

    # Saves relevant bib entries along with their citation
    print("Getting info for bibrefs")
    for i in trange(0, len(all_bib_hashes), BATCH_SIZE):
        bib_hashes_batch = all_bib_hashes[i : i + BATCH_SIZE]
        bib_entry_raw_batch = [all_bib_entries[bib_hash]["bib_entry_raw"] for bib_hash in bib_hashes_batch]
        titles_batch = get_titles_s2_internal(bib_entry_raw_batch)
        corpus_ids_batch = get_corpus_ids_s2_internal(titles_batch)
        metadata_batch = get_metadata_s2_public(corpus_ids_batch)

        bib_hash_dict = [
            {"bib_hash_or_arxiv_id": bib_hash, "title": title, "corpus_id": corpus_id, "metadata": metadata}
            | all_bib_entries[bib_hash]
            for bib_hash, title, corpus_id, metadata in zip(
                bib_hashes_batch, titles_batch, corpus_ids_batch, metadata_batch
            )
        ]

        with open(args.out_path, "a") as f:
            for line in bib_hash_dict:
                f.write(json.dumps(line) + "\n")

        time.sleep(0.5)  # for rate-limiting
    print("Getting info for arxiv ids")
    # Next, do the arxiv ids
    for i in trange(0, len(all_arxiv_ids), BATCH_SIZE):
        arxiv_ids_batch = all_arxiv_ids[i : i + BATCH_SIZE]
        # remove the versioning information
        arxiv_ids_batch_stripped = [re.sub("v\d+", "", arxiv_id) for arxiv_id in arxiv_ids_batch]
        metadata_batch = get_metadata_s2_public(arxiv_ids_batch_stripped, prefix="ARXIV:")

        # Kind of overloading this a bit... because we don't have bib_hashes for the ArXiv
        # ones, so let's instead store the ArXiv id there. (Could also be the paper hash tbh,
        # but that might be more confusing. We also shouldn't be *using* the bib_hash anywhere
        # so it's probably ok)
        bib_hash_dict = [
            {
                "bib_hash_or_arxiv_id": arxiv_id,
                "title": metadata["title"],
                "corpus_id": metadata.get("corpusId", -1),
                "metadata": metadata,
            }
            for arxiv_id, metadata in zip(arxiv_ids_batch, metadata_batch)
        ]
        with open(args.out_path, "a") as f:
            for line in bib_hash_dict:
                f.write(json.dumps(line) + "\n")

        time.sleep(0.5)  # for rate-limiting

    # Add the corpus ids to the dataset file
    print("Editing the dataset file to add the corpus ids... Do not press CTRL-C")

    with open(args.out_path) as f:
        bib_hash_dict = [json.loads(line) for line in f]
        bib_entry_map = {entry["bib_hash_or_arxiv_id"]: entry for entry in bib_hash_dict}

    new_dataset = []
    with open(args.dataset_path) as f:
        for line in f:
            sample = json.loads(line)
            for row in sample["row_bib_map"]:
                row["corpus_id"] = bib_entry_map[row["bib_hash_or_arxiv_id"]]["corpus_id"]
            new_dataset.append(sample)

    with open(args.dataset_path, "w") as f:
        for sample in new_dataset:
            f.write(json.dumps(sample) + "\n")
    print("... Done")


if __name__ == "__main__":
    main()
