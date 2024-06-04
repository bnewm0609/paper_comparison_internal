import sys
import os
import json

from paper_comparison.types import Table
from paper_comparison.metrics_utils import DecontextFeaturizer

gold_table_file = open(sys.argv[1])
base_folder = sys.argv[2]
gold_table_list = [json.loads(x) for x in gold_table_file.readlines()]
featurizer = DecontextFeaturizer("decontext")

for i, table in enumerate(gold_table_list):
    print(f"Processing table {i}")
    print(table["tabid"])
    if os.path.exists(os.path.join(base_folder, f'{table["tabid"]}_gold.json')):
        continue
    table_instance = Table(table["tabid"], list(table["table"].keys()), table["table"])
    decontext_columns = featurizer.featurize(table_instance.schema, table_instance)
    table['decontext_schema'] = dict(zip(table_instance.schema, decontext_columns))
    out_file = open(os.path.join(base_folder, f'{table["tabid"]}_gold.json'), 'w')
    json.dump(table, out_file)
    out_file.close()