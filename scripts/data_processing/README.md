# Data Processing Pipeline

I've been scraping tables from arxiv publications one month at a time, using the following steps:

1. Download all of the arxiv latex (on the s2 cluster, in the `arxiv_tables` directory):
```
aws s3 cp --recursive "s3://ai2-s2-scholarphi-pipeline-prod/daq/arxiv-source-data/bymonth/yymm" in_latex_s3/yymm
```

For example:
```
aws s3 cp --recursive "s3://ai2-s2-scholarphi-pipeline-prod/daq/arxiv-source-data/bymonth/2310" in_latex_s3/2310
```

2. This gives a directory with a bunch of `*.gz` files in it, but for the table extraction step, we need a single tar file, so we create one:
```
tar cvf in_tar/2310.00000-07773.tar in_latex_s3/2310/
```
By convention, I'm using the range of papers posted on arxiv that month as the tar file name. This is because sometimes the month gets split up, or for debugging purposes I'm considering a subset of the papers, and in those situations its useful to mark that in the tar file name.

3. Extract the table and bibliography xml from the latex files. This takes a bit of set-up before running the first time, as outlined here: [](https://github.com/bnewm0609/unarXive/tree/master/src). First install the required software:
```
sudo apt-get update
sudo apt install texlive-extra-utils
sudo apt install tralics
```

Then clone my fork of the repo and run steps 1 and 3. I only followed steps 1 and 3 because I'm using S2 apis to do the citation matching.
```
python unarXive/src/prepare.py in_tar/ out_xml/ arxiv-metadata-oai-snapshot.sqlite
```

4. Now, I move the generated data into a directory called `arxiv_dump/out_xml` on my local machine (you don't have to do this, but I did because I was worried about losing access to the cluster).
```
scp benjaminn@s2-cirrascale-10.reviz.ai2.in:~/nfs/arxiv_tables/out_xml/2310.00000-07773.jsonl arxiv_dump/out_xml/
```

5. Then, I filter the tables [not tested yet]:
```
python scripts/data_processing/extract_tables.py arxiv_dump/out_xml/2310.00000-07773.jsonl arxiv_dump/out_xml_filtered/2310.00000-07773_dataset.jsonl
```

6. Then, I match the citations to the s2 database, which requires being on VPN because it calls s2 internal apis:
```
python scripts/data_processing/populate_bib_entities.py ../arxiv_dump/out_xml/2308.00000-16912v1.jsonl ../arxiv_dump/out_xml_filtered/2308.00000-16912v1_dataset_NO_FLOATS_has_cites_has_max_2_subtables_has_2_cols_2_rows_not_long.jsonl ../arxiv_dump/out_bib_entries.jsonl
```
7. Then, I run this short script to create the stripped-down datasets for the code [not tested yet]
```
python scripts/data_processing/create_tables_and_papers_datasets.py in_tables_path out_tables_path.jsonl out_papers_path.jsonl
```

For example,
```
python scripts/data_processing/create_tables_and_papers_datasets.py ../arxiv_dump/out_xml_filtered/2308.00000-16912v1_dataset_NO_FLOATS_has_cites_has_max_2_subtables_has_2_cols_2_rows_not_long.jsonl paper_comparison/data/arxiv_tables/2308_tables.jsonl paper_comparison/data/arxiv_tables/2308_papers.jsonl
```

8. Finally, I download the full texts from athena [not tested yet]
```
python scripts/data_processing/download_full_texts.py data/arxiv_tables/2308_papers.jsonl
```