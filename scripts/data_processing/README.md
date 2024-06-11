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
By convention, I've been using the range of papers posted on arxiv that month as the tar file name. This is in case the month gets split up, or for debugging purposes I'm considering a subset of the papers, and in those situations its useful to mark that in the tar file name.



3. Extract the table and bibliography xml from the latex files. This takes a bit of set-up before running the first time, as outlined here: [](https://github.com/bnewm0609/unarXive/tree/master/src). First install the required software:
```
sudo apt-get update
yes | sudo apt install texlive-extra-utils
yes | sudo apt install tralics
```

Then clone my fork of the [`unarxiv`](https://github.com/bnewm0609/unarXive/tree/master/src) repo and run steps 1 and 3. I only followed steps 1 and 3 because I'm using S2 apis to do the citation matching.
```
python unarXive/src/prepare.py in_tar/ out_xml/ arxiv-metadata-oai-snapshot.sqlite
```

4. Now, move the generated data into a directory called `arxiv_dump/out_xml` on my local machine (you don't have to do this, but I did because I was worried about losing access to the cluster).
```
scp benjaminn@s2-cirrascale-10.reviz.ai2.in:~/nfs/arxiv_tables/out_xml/2310.00000-07773.jsonl arxiv_dump/out_xml_filtered/2310.00000-07773_filtered.jsonl
```

5. Then, filter the tables and convert them to pandas format:
```
python scripts/data_processing/extract_tables.py arxiv_dump/out_xml/2310.00000-07773.jsonl arxiv_dump/out_xml_filtered/2310.00000-07773_dataset.jsonl
```
You can add the `--check_yield` option to print out the number of tables after the filtering step without actually trying to convert it to pandas format or saving the data.

For providing labels:
`# python scripts/data_processing/extract_tables.py ../out_xml_fulltext/ --out_labeled_path ../out_xml_fulltext_labeled/ --label_only --num_processes 8`
`python scripts/data_processing/extract_tables.py ../out_xml_fulltext/ --out_labeled_path ../out_xml_fulltext_labeled/ --out_filtered_path ../out_xml_fulltext_filtered/valid_tables_with_floats_and_figures/ --label --filter --num_processes 4`

For actually doing the filtering:
`python scripts/data_processing/extract_tables.py ../arxiv_dump/out_xml_fulltext_labeled/2212.jsonl.gz --out_filtered_path ../arxiv_dump/out_xml_fulltext_filtered/valid_tables_with_floats_and_figures/2212.jsonl --filter`

For doing post-filtering:
`python scripts/data_processing/extract_tables.py ../arxiv_dump/out_xml_fulltext_filtered/valid_tables_with_floats_and_figures_json.jsonl --create_quality_datasets --out_high_quality_path data/arxiv_tables_with_floats_and_figures/high_quality_tables.jsonl --out_high_quality_schemes_path data/arxiv_tables_with_floats_and_figures/high_quality_schemes.jsonl --out_mid_quality_path data/arxiv_tables_with_floats_and_figures/medium_quality_tables.jsonl`

6. Then, to match the citations to the s2 database, which requires being on VPN because it calls s2 internal apis:
```
python scripts/data_processing/populate_bib_entries.py ../arxiv_dump/out_xml/2308.00000-16912v1.jsonl ../arxiv_dump/out_xml_filtered/2308.00000-16912v1_dataset_NO_FLOATS_has_cites_has_max_2_subtables_has_2_cols_2_rows_not_long.jsonl ../arxiv_dump/out_bib_entries.jsonl
```

7. Then download the full texts from s2orc using athena
```
python scripts/data_processing/download_full_texts.py data/arxiv_tables/2308_papers.jsonl
```

8. Then, download and unarxiv just the full texts associated with the papers.

9. Then, run this short script to create the stripped-down datasets for the code
```
python scripts/data_processing/create_tables_and_papers_datasets.py in_tables_path out_tables_path.jsonl out_papers_path.jsonl
```

For example,
```
python scripts/data_processing/create_tables_and_papers_datasets.py ../arxiv_dump/out_xml_filtered/2308.00000-16912v1_dataset_NO_FLOATS_has_cites_has_max_2_subtables_has_2_cols_2_rows_not_long.jsonl paper_comparison/data/arxiv_tables/2308_tables.jsonl paper_comparison/data/arxiv_tables/2308_papers.jsonl
```

# Saving Fulltext XML to S3

1. Run unarxiv with the full text:
```
python unarXive/src/prepare.py in_tar/ out_xml_fulltext/ arxiv-metadata-oai-snapshot.sqlite --parse_fulltext --tralics_dir out_tralics_fulltext/2308/
```

2. Compress the individual xml files

3. Upload to s3


# Manually editing tables:

```
python scripts/data_processing/data_editor.py data/v3/highest_quality_tables_1k/dataset.jsonl <tab_id> --out_file data/xml_to_json_gold_data/dataset.jsonl
```