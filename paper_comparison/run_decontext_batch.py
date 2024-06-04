import sys
import os
import json

from paper_comparison.types import Table
from paper_comparison.metrics_utils import DecontextFeaturizer

base_folder = sys.argv[1]
model_name = sys.argv[2]
variant_name = sys.argv[3]

featurizer = DecontextFeaturizer("decontext")

for i, file in enumerate(os.listdir(base_folder)):
    print(f"Processing file {i}")
    if not os.path.isdir(os.path.join(base_folder, file)):
        continue
    if not os.path.exists(os.path.join(base_folder, file, model_name, f'{variant_name}_outputs', 'try_0.json')):
        continue
    if os.path.exists(os.path.join(base_folder, file, model_name, f'{variant_name}_outputs', 'try_0_decontext.json')):
        continue
    table_file = os.path.join(base_folder, file, model_name, f'{variant_name}_outputs', 'try_0.json')
    table_data = json.loads(open(table_file).read())
    for table in table_data:
        table_instance = Table(table["tabid"], table["schema"], table["table"])
        decontext_columns = featurizer.featurize(table_instance.schema, table_instance)
        table['decontext_schema'] = dict(zip(table_instance.schema, decontext_columns))
    out_file = open(os.path.join(base_folder, file, model_name, f'{variant_name}_outputs', 'try_0_decontext.json'), 'w')
    json.dump(table_data, out_file)
    out_file.close()
    