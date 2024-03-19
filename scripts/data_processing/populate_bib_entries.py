from argparse import ArgumentParser
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import re
import requests_cache
import requests
from tqdm import tqdm
from nltk import word_tokenize, edit_distance

# import requests as session
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
    headers = {"x-api-key": "DGuWvVeW6K4DSzIOmbIg190V3sOXias928otghb2"}
    response = session.post(base_url, json=ids_json, headers=headers)
    metadata = response.json()

    try:
        if metadata.get("code") == "429":
            print("Unable to get metadata due to s2_public rate limit")
            metadata = [None for _ in range(len(corpus_ids_batch))]
        elif metadata.get("error") == "No valid paper ids given":
            print("No valid paper ids given")
            metadata = [None for _ in range(len(corpus_ids_batch))]
    except AttributeError:
        pass

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
    if os.path.isdir(args.papers_path):
        papers_paths = [os.path.join(args.papers_path, fn) for fn in os.listdir(args.paper_path)]
    else:
        papers_paths = [args.papers_path]

    for papers_path in papers_paths:
        valid_tables = []
        if os.path.splitext(papers_path)[1] == ".gz":
            f = gzip.open(papers_path, "r")
        else:
            f = open(papers_path)
        # with open(args.papers_path) as f:
        for line in f:
            paper = json.loads(line)
            # Extracts the bib_entities
            all_bib_entries |= paper["bib_entries"]
        f.close()

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
    # breakpoint()
    print("Getting info for bibrefs")
    bib_hash_dict = []
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

    if not bib_hash_dict:
        print("No valid tables!")
    else:
        print(bib_hash_dict[0])
    print("Done!")
    # print("Getting info for arxiv ids")
    # # Next, do the arxiv ids
    # breakpoint()
    # for i in trange(0, len(all_arxiv_ids), BATCH_SIZE):
    #     arxiv_ids_batch = all_arxiv_ids[i : i + BATCH_SIZE]
    #     # remove the versioning information
    #     arxiv_ids_batch_stripped = [re.sub("v\d+", "", arxiv_id) for arxiv_id in arxiv_ids_batch]
    #     metadata_batch = get_metadata_s2_public(arxiv_ids_batch_stripped, prefix="ARXIV:")

    #     # Kind of overloading this a bit... because we don't have bib_hashes for the ArXiv
    #     # ones, so let's instead store the ArXiv id there. (Could also be the paper hash tbh,
    #     # but that might be more confusing. We also shouldn't be *using* the bib_hash anywhere
    #     # so it's probably ok)
    #     bib_hash_dict = [
    #         {
    #             "bib_hash_or_arxiv_id": arxiv_id,
    #             "title": metadata["title"],
    #             "corpus_id": metadata.get("corpusId", -1),
    #             "metadata": metadata,
    #         }
    #         for arxiv_id, metadata in zip(arxiv_ids_batch, metadata_batch)
    #     ]
    #     with open(args.out_path, "a") as f:
    #         for line in bib_hash_dict:
    #             f.write(json.dumps(line) + "\n")

    #     time.sleep(0.5)  # for rate-limiting

    # # Add the corpus ids to the dataset file
    # print("Editing the dataset file to add the corpus ids... Do not press CTRL-C")

    # with open(args.out_path) as f:
    #     bib_hash_dict = [json.loads(line) for line in f]
    #     bib_entry_map = {entry["bib_hash_or_arxiv_id"]: entry for entry in bib_hash_dict}

    # new_dataset = []
    # with open(args.dataset_path) as f:
    #     for line in f:
    #         sample = json.loads(line)
    #         for row in sample["row_bib_map"]:
    #             row["corpus_id"] = bib_entry_map[row["bib_hash_or_arxiv_id"]]["corpus_id"]
    #         new_dataset.append(sample)

    # with open(args.dataset_path, "w") as f:
    #     for sample in new_dataset:
    #         f.write(json.dumps(sample) + "\n")
    # print("... Done")


def normalize(text):
    # remove punctuation
    text = re.sub("{{formula.+?}}", "", text)
    text = re.sub("\S+", lambda m: re.sub("^\W+|\W+$", "", m.group()), text)
    # remove surrounding quotes and spaces
    text = text.strip('"')
    text = text.strip()
    # remove any formulas
    return text


def get_corpus_ids_and_metadata_s2_public(titles_batch, verbose=False):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"fields": "corpusId,externalIds,title,abstract", "limit": 5}
    headers = {"x-api-key": "DGuWvVeW6K4DSzIOmbIg190V3sOXias928otghb2"}
    outputs = []
    responses = []
    for title in titles_batch:
        if "{{formula}}" in title:
            outputs.append(None)
            responses.append(None)
            continue

        title_normalized = normalize(title)
        params["query"] = title
        # response = requests.get(base_url, params)
        response = session.get(base_url, params, headers=headers)
        response = response.json()
        responses.append(response)

        if verbose:
            print(response)
        if response.get("data") is None or response.get("total") == 0:
            outputs.append(None)
            continue

        if verbose:
            print(title_normalized)
        for output in response.get("data"):
            output_title_normalized = normalize(output["title"])
            if verbose:
                print(output_title_normalized)
            if (
                output_title_normalized in title
                or title in output_title_normalized
                or edit_distance(word_tokenize(output_title_normalized), word_tokenize(title_normalized)) < 4
            ):
                outputs.append(output)
                break
        else:
            outputs.append(None)
        if len(titles_batch) > 1:
            time.sleep(1)
    return outputs, responses


def uniql(lst):
    lst_uniq = []
    for item in lst:
        if item not in lst_uniq:
            lst_uniq.append(item)
    return lst_uniq


def main_2():
    """
    Hits the public API to search for papers that we don't have corpus ids for. This doesn't really work
    in that we only are able to get ~5k of the corpus ids, but we save the responses to see if we
    can match based on any other heuristics at a later date.
    """
    print("Loading in our out_bib_entries file...", end="", flush=True)
    with open("arxiv_dump/out_bib_entries.jsonl") as f:
        out_bib_entries = [json.loads(line) for line in f]
        out_bib_entries = {entry["bib_hash_or_arxiv_id"]: entry for entry in out_bib_entries}
    print("done.")

    missing_corpus_ids = []
    for bib_hash in out_bib_entries:
        if out_bib_entries[bib_hash]["corpus_id"] == -1:
            missing_corpus_ids.append(bib_hash)
    missing_corpus_ids = uniql(missing_corpus_ids)

    if os.path.exists("arxiv_dump/out_bib_entries/s2_pub_search_selected.jsonl"):
        with open("arxiv_dump/out_bib_entries/s2_pub_search_selected.jsonl") as f:
            all_outputs = [json.loads(line) for line in f]
            all_outputs = {resp["bib_hash"]: resp for resp in all_outputs}
    else:
        all_outputs = {}
    all_responses = {}
    # breakpoint()

    num_preprocessed_missing = 0
    num_preprocessed_obtained = 0
    num_errors_or_timeouts = 0
    printed_previous_summary = False
    unsaved_bib_hashes = []
    for i, bib_hash in tqdm(enumerate(missing_corpus_ids), total=len(missing_corpus_ids)):
        # if i > 16:
        #     break
        if out_bib_entries[bib_hash]["title"] is None:
            continue

        if bib_hash in all_outputs and all_outputs[bib_hash]["corpusId"] == -1:
            # we processed this one already, and we didn't get anything
            # print("Skipping because we didn't find anything before")
            num_preprocessed_missing += 1
            continue

        elif bib_hash in all_outputs and out_bib_entries[bib_hash]["corpus_id"] == -1:
            # we processed this one already, but there was a bug, so we didn't save the changes - just update with what we already have
            # print("Skipping because we processed this one already")
            num_preprocessed_obtained += 1
            out_bib_entries[bib_hash]["corpus_id"] = all_outputs[bib_hash]["corpusId"]
            out_bib_entries[bib_hash]["metadata"] = all_outputs[bib_hash]
            continue

        elif bib_hash in all_outputs:
            # we processed this bib_hash already, but already saved the output in out_bib_entries, so we should skip it
            continue

        title = out_bib_entries[bib_hash]["title"]

        outputs, responses = get_corpus_ids_and_metadata_s2_public([title], verbose=False)
        if responses[0].get("message") in {"Internal Server Error", "Endpoint request timed out"}:
            # skip for now
            num_errors_or_timeouts += 1
            if num_errors_or_timeouts % 100 == 0:
                print(f"Skipped {num_errors_or_timeouts} so far.")
            continue

        all_responses[bib_hash] = responses[0]
        all_responses[bib_hash]["bib_hash"] = bib_hash

        if outputs[0] is not None:
            all_outputs[bib_hash] = outputs[0]
            all_outputs[bib_hash]["bib_hash"] = bib_hash

            out_bib_entries[bib_hash]["corpus_id"] = outputs[0]["corpusId"]
            out_bib_entries[bib_hash]["metadata"] = outputs[0]
        else:
            all_outputs[bib_hash] = {"bib_hash": bib_hash, "corpusId": -1}

        unsaved_bib_hashes.append(bib_hash)
        if len(unsaved_bib_hashes) >= 100:
            print("Saving responses...", end="", flush=True)
            for unsaved_bib_hash in unsaved_bib_hashes:
                with open("arxiv_dump/out_bib_entries/s2_pub_search_responses.jsonl", "a") as f:
                    f.write(json.dumps(all_responses[unsaved_bib_hash]) + "\n")
                with open("arxiv_dump/out_bib_entries/s2_pub_search_selected.jsonl", "a") as f:
                    f.write(json.dumps(all_outputs[unsaved_bib_hash]) + "\n")
            print(" done")
            unsaved_bib_hashes = []
        time.sleep(1)

    if len(unsaved_bib_hashes) >= 0:
        print("Saving responses...", end="", flush=True)
        for unsaved_bib_hash in unsaved_bib_hashes:
            with open("arxiv_dump/out_bib_entries/s2_pub_search_responses.jsonl", "a") as f:
                f.write(json.dumps(all_responses[unsaved_bib_hash]) + "\n")
            with open("arxiv_dump/out_bib_entries/s2_pub_search_selected.jsonl", "a") as f:
                f.write(json.dumps(all_outputs[unsaved_bib_hash]) + "\n")
        print(" done")

    print(f"Number of errors or timeouts: {num_errors_or_timeouts}. To get these, please re-run the script")
    os.rename("arxiv_dump/out_bib_entries.jsonl", "arxiv_dump/out_bib_entries/out_bib_entries.jsonl.bak6")
    with open("arxiv_dump/out_bib_entries.jsonl", "w") as f:
        for key in out_bib_entries:
            f.write(json.dumps(out_bib_entries[key]) + "\n")

    if not printed_previous_summary:
        print(f"Skipped because we didn't find anything: {num_preprocessed_missing}")
        print(f"Skipped because we already processed: {num_preprocessed_obtained}")
        printed_previous_summary = True


def main_3():
    """
    Hit the /batch end point to get paper details for a bunch of papers that we have
    corpus ids for but no metadata (yet)
    """
    print("Loading in our out_bib_entries file...", end="", flush=True)
    with open("arxiv_dump/out_bib_entries.jsonl") as f:
        out_bib_entries = [json.loads(line) for line in f]
        out_bib_entries = {entry["bib_hash_or_arxiv_id"]: entry for entry in out_bib_entries}
    print("done.")

    corpus_ids_missing_metadata = defaultdict(list)
    for bib_hash in out_bib_entries:
        if out_bib_entries[bib_hash]["corpus_id"] != -1 and (
            out_bib_entries[bib_hash]["metadata"] is None
            or isinstance(out_bib_entries[bib_hash]["metadata"], str)
            or "abstract" not in out_bib_entries[bib_hash]["metadata"]
        ):
            corpus_ids_missing_metadata[out_bib_entries[bib_hash]["corpus_id"]].append(bib_hash)

    print(len(corpus_ids_missing_metadata))

    # import sys

    # sys.exit(0)
    corpus_ids_missing_metadata_list = list(corpus_ids_missing_metadata.keys())
    for i in trange(0, len(corpus_ids_missing_metadata_list), BATCH_SIZE):
        # bib_ref_batch = [t[0] for t in corpus_ids_missing_metadata[i : i + BATCH_SIZE]]
        corpus_ids_batch = corpus_ids_missing_metadata_list[i : i + BATCH_SIZE]
        metadata_batch = get_metadata_s2_public(corpus_ids_batch)

        for corpus_id, metadata in zip(corpus_ids_batch, metadata_batch):
            for bib_hash in corpus_ids_missing_metadata[corpus_id]:
                if (
                    out_bib_entries[bib_hash]["metadata"] is None
                    or isinstance(out_bib_entries[bib_hash]["metadata"], str)
                    or "abstract" not in out_bib_entries[bib_hash]["metadata"]
                ):
                    out_bib_entries[bib_hash]["metadata"] = metadata
        time.sleep(1)

    # get_metadata_s2_public(corpus_ids_batch)
    print("Saving...")
    with open("arxiv_dump/out_bib_entries.jsonl", "w") as f:
        for key in out_bib_entries:
            f.write(json.dumps(out_bib_entries[key]) + "\n")
    print("Done")


if __name__ == "__main__":
    # main()
    # main_2()
    main_3()
