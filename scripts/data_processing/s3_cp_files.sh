#!/bin/bash
# Example usage: sh s3_cp_files.sh 2308 2308_high_quality.txt 2308_high_quality
set -e
while read line
do
  aws s3 cp s3://ai2-s2-scholarphi-pipeline-prod/daq/arxiv-source-data/bymonth/$1/$line $3
done <$2