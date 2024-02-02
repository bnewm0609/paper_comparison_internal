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

import boto3


PERM_OUT_DIR_NAME = "data/full_texts"
OUT_DIR_NAME = "tmp"


def download_s2orc(corpus_ids, aws_folder="s3://ai2-s2-benjaminn/tmp_arxiv_tables"):
    # # Clean up temporary folder
    os.system(f"aws s3 rm --recursive {aws_folder}")
    shutil.rmtree("tmp", ignore_errors=True)

    # The way this *should* work is:
    #  1. Create an external table that contains the corpus ids
    #  2. Join against that table
    # I don't know AWS Athena or SQL that well, so I couldn't easily figure
    # this out. So instead, my query uses an "IN" statement, which is fine for
    # small numbers of tables, but might not be fine for thousands.

    # # # Not sure how to create external tables, but this is how it should be done...
    # # # create a new temp table with the corpus ids
    # # s3 = boto3.resource("s3")
    # # object = s3.Object("ai2-s2-benjaminn", "tmp_corpus_ids/tmp_corpus_ids.csv")
    # # corpus_id_data = "\n".join(["corpus_id"] + corpus_ids)
    # # object.put(Body=corpus_id_data)

    # # session = boto3.Session()
    # # athena = session.client("athena", "us-west-2")
    # # external_table_query = """\
    # # CREATE EXTERNAL TABLE IF NOT EXISTS
    # # s2orc_papers.tmp_benjaminn_corpus_ids (
    # #     corpus_id string
    # # ) ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
    # # WITH SERDEPROPERTIES (
    # # 'separatorChar' = ',',
    # # 'quoteChar' = '\"',
    # # 'escapeChar' = '\\'
    # # )
    # # STORED AS TEXTFILE
    # # LOCATION 's3://ai2-s2-benjaminn/tmp_corpus_ids/';
    # # """

    session = boto3.Session()
    athena = session.client("athena", "us-west-2")
    # # execution = athena.start_query_execution(
    # #     QueryString=query,
    # #     QueryExecutionContext={"Database": "s2orc_papers"},
    # #     ResultConfiguration={
    # #         "OutputLocation": aws_folder,
    # #     },
    # # )

    query = f"""
    UNLOAD
    (
        SELECT *
        FROM s2orc_papers.latest
        WHERE id IN ({','.join(corpus_ids)})
    )
    TO '{aws_folder}'
    WITH (format='json', compression='gzip')
    """

    execution = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": "s2orc_papers"},
        ResultConfiguration={
            "OutputLocation": aws_folder,
        },
    )

    execution_id = execution["QueryExecutionId"]
    max_execution = 60

    state = "RUNNING"
    while max_execution > 0 and state in ["RUNNING", "QUEUED"]:
        max_execution = max_execution - 1
        response = athena.get_query_execution(QueryExecutionId=execution_id)

        if (
            "QueryExecution" in response
            and "Status" in response["QueryExecution"]
            and "State" in response["QueryExecution"]["Status"]
        ):
            state = response["QueryExecution"]["Status"]["State"]
            if state == "FAILED":
                print("FAILED", response)
            elif state == "SUCCEEDED":
                s3_path = response["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
                filename = re.findall(".*\/(.*)", s3_path)[0]
                print("SUCCEEDED", filename)
        time.sleep(1)
    # breakpoint()
    # os.makedirs("tmp", exist_ok=True)
    os.system(f"aws s3 sync {aws_folder} {OUT_DIR_NAME}")
    index_file = glob.glob(f"{OUT_DIR_NAME}/*.csv")[0]
    s2orc = []
    with open(index_file) as filelist:
        files = csv.reader(filelist)
        for f in files:
            filename = f[0].split("/")[-1]
            with gzip.open(f"{OUT_DIR_NAME}/{filename}", mode="rt") as _f:
                for line in _f:
                    s2orc.append(json.loads(line))
    return s2orc


def main():
    argp = ArgumentParser()
    argp.add_argument("in_path", type=str, help="bib_entries file to get the full texts for")
    argp.add_argument("--out_dir", type=str, help="", default=PERM_OUT_DIR_NAME)
    args = argp.parse_args()

    with open(args.in_path) as f:
        bib_entries = [json.loads(line.strip()) for line in f]
        corpus_ids = [entry.get("corpus_id") for entry in bib_entries if entry.get("corpus_id") is not None]

    os.makedirs(OUT_DIR_NAME, exist_ok=True)
    os.makedirs(argp.out_dir, exist_ok=True)
    downloaded_texts = [os.path.splitext(fn)[0] for fn in os.listdir(argp.out_dir)]
    corpus_ids = [str(corpus_id) for corpus_id in corpus_ids if str(corpus_id) not in downloaded_texts]

    # print(len(downloaded_texts))
    # print(corpus_ids)
    print(len(corpus_ids))
    s2orc = download_s2orc(corpus_ids)

    for paper in s2orc:
        with open(Path(argp.out_dir) / f"{paper['id']}.json", "w") as f:
            json.dump(paper, f)


if __name__ == "__main__":
    main()
