from argparse import ArgumentParser
import csv
import glob
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import re
from tqdm import tqdm

# import boto3


# PERM_OUT_DIR_NAME = "data/full_texts"
# OUT_DIR_NAME = "tmp"


# def download_s2orc(corpus_ids, aws_folder="s3://ai2-s2-benjaminn/tmp_arxiv_tables"):
#     # # Clean up temporary folder
#     os.system(f"aws s3 rm --recursive {aws_folder}")
#     shutil.rmtree("tmp", ignore_errors=True)

#     # The way this *should* work is:
#     #  1. Create an external table that contains the corpus ids
#     #  2. Join against that table
#     # I don't know AWS Athena or SQL that well, so I couldn't easily figure
#     # this out. So instead, my query uses an "IN" statement, which is fine for
#     # small numbers of tables, but might not be fine for thousands.

#     # # # Not sure how to create external tables, but this is how it should be done...
#     # # # create a new temp table with the corpus ids
#     # # s3 = boto3.resource("s3")
#     # # object = s3.Object("ai2-s2-benjaminn", "tmp_corpus_ids/tmp_corpus_ids.csv")
#     # # corpus_id_data = "\n".join(["corpus_id"] + corpus_ids)
#     # # object.put(Body=corpus_id_data)

#     # # session = boto3.Session()
#     # # athena = session.client("athena", "us-west-2")
#     # # external_table_query = """\
#     # # CREATE EXTERNAL TABLE IF NOT EXISTS
#     # # s2orc_papers.tmp_benjaminn_corpus_ids (
#     # #     corpus_id string
#     # # ) ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
#     # # WITH SERDEPROPERTIES (
#     # # 'separatorChar' = ',',
#     # # 'quoteChar' = '\"',
#     # # 'escapeChar' = '\\'
#     # # )
#     # # STORED AS TEXTFILE
#     # # LOCATION 's3://ai2-s2-benjaminn/tmp_corpus_ids/';
#     # # """

#     session = boto3.Session()
#     athena = session.client("athena", "us-west-2")
#     # # execution = athena.start_query_execution(
#     # #     QueryString=query,
#     # #     QueryExecutionContext={"Database": "s2orc_papers"},
#     # #     ResultConfiguration={
#     # #         "OutputLocation": aws_folder,
#     # #     },
#     # # )

#     query = f"""
#     UNLOAD
#     (
#         SELECT *
#         FROM s2orc_papers.latest
#         WHERE id IN ({','.join(corpus_ids)})
#     )
#     TO '{aws_folder}'
#     WITH (format='json', compression='gzip')
#     """

#     execution = athena.start_query_execution(
#         QueryString=query,
#         QueryExecutionContext={"Database": "s2orc_papers"},
#         ResultConfiguration={
#             "OutputLocation": aws_folder,
#         },
#     )

#     execution_id = execution["QueryExecutionId"]
#     max_execution = 60

#     state = "RUNNING"
#     while max_execution > 0 and state in ["RUNNING", "QUEUED"]:
#         max_execution = max_execution - 1
#         response = athena.get_query_execution(QueryExecutionId=execution_id)

#         if (
#             "QueryExecution" in response
#             and "Status" in response["QueryExecution"]
#             and "State" in response["QueryExecution"]["Status"]
#         ):
#             state = response["QueryExecution"]["Status"]["State"]
#             if state == "FAILED":
#                 print("FAILED", response)
#             elif state == "SUCCEEDED":
#                 s3_path = response["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
#                 filename = re.findall(".*\/(.*)", s3_path)[0]
#                 print("SUCCEEDED", filename)
#         time.sleep(1)
#     # breakpoint()
#     # os.makedirs("tmp", exist_ok=True)
#     os.system(f"aws s3 sync {aws_folder} {OUT_DIR_NAME}")
#     index_file = glob.glob(f"{OUT_DIR_NAME}/*.csv")[0]
#     s2orc = []
#     with open(index_file) as filelist:
#         files = csv.reader(filelist)
#         for f in files:
#             filename = f[0].split("/")[-1]
#             with gzip.open(f"{OUT_DIR_NAME}/{filename}", mode="rt") as _f:
#                 for line in _f:
#                     s2orc.append(json.loads(line))
#     return s2orc


# def main():
#     argp = ArgumentParser()
#     argp.add_argument("in_path", type=str, help="bib_entries file to get the full texts for")
#     argp.add_argument("--out_dir", type=str, help="", default=PERM_OUT_DIR_NAME)
#     args = argp.parse_args()

#     with open(args.in_path) as f:
#         bib_entries = [json.loads(line.strip()) for line in f]
#         corpus_ids = [entry.get("corpus_id") for entry in bib_entries if entry.get("corpus_id") is not None]

#     os.makedirs(OUT_DIR_NAME, exist_ok=True)
#     os.makedirs(argp.out_dir, exist_ok=True)
#     downloaded_texts = [os.path.splitext(fn)[0] for fn in os.listdir(argp.out_dir)]
#     corpus_ids = [str(corpus_id) for corpus_id in corpus_ids if str(corpus_id) not in downloaded_texts]

#     # print(len(downloaded_texts))
#     # print(corpus_ids)
#     print(len(corpus_ids))
#     s2orc = download_s2orc(corpus_ids)

#     for paper in s2orc:
#         with open(Path(argp.out_dir) / f"{paper['id']}.json", "w") as f:
#             json.dump(paper, f)

import time
import requests


def save_jsons(jsons, out_file):
    with open(out_file, "a") as f:
        for sample in jsons:
            f.write(json.dumps(sample) + "\n")


def main_2():
    argp = ArgumentParser()
    argp.add_argument("papers_file")
    argp.add_argument("--out_file")
    argp.add_argument("--start", type=int, default=0)
    argp.add_argument("--count", type=int, default=None)
    args = argp.parse_args()
    print("starting")

    data_jsons = []
    with open(args.papers_file) as f:
        papers = [json.loads(line) for line in f]

    start = args.start
    count = len(papers) - args.start if args.count is None else args.count
    papers = papers[start : start + count]

    # filter out corpus_ids that we've already downloaded
    try:
        with open(args.out_file) as f:
            obtained_corpus_ids = [json.loads(line)["metadata"]["corpusId"] for line in tqdm(f, total=2513)]
            # obtained_corpus_ids = {paper["metadata"]["corpusId"] for paper in previous_papers}
            assert obtained_corpus_ids
            print(list(obtained_corpus_ids[:10]), len(obtained_corpus_ids))

            papers = [paper for paper in papers if paper["corpus_id"] not in obtained_corpus_ids]
    except FileNotFoundError:
        pass

    unshowable_corpus_ids = []
    other_error_corpus_ids = []
    for sample in tqdm(papers):
        corpus_id = sample["corpus_id"]

        response = requests.get(f"https://mage.allen.ai/document/{corpus_id}")

        data = response.json()

        if "error" in data:
            # print(f"Skipping {corpus_id} due to error:")
            # print(data)
            if isinstance(data["error"], str) and data["error"].startswith("CorpusId is not showable"):
                unshowable_corpus_ids.append(corpus_id)
            else:
                other_error_corpus_ids.append(corpus_id)

            time.sleep(0.1)
            continue

        # print(data.keys())

        if "metadata" not in data:
            print(data.keys())
            data["metadata"] = {"corpusId": corpus_id}
        else:
            data["metadata"]["corpusId"] = corpus_id

        data_jsons.append(data)
        if len(data_jsons) >= 5:
            save_jsons(data_jsons, args.out_file)
            data_jsons = []

        time.sleep(0.5)

    save_jsons(data_jsons, args.out_file)
    save_jsons(
        [{"unshowable_ids": unshowable_corpus_ids, "other_error_ids": other_error_corpus_ids}],
        os.path.splitext(args.out_file)[0] + "_errors.jsonl",
    )
    print("done")


if __name__ == "__main__":
    # main()
    main_2()
