from paper_comparison.types import Table
from paper_comparison.metrics_utils import JaccardAlignmentScorer, ExactMatchScorer, EditDistanceScorer, SentenceTransformerAlignmentScorer
from paper_comparison.metrics_utils import BaseFeaturizer, ValueFeaturizer, DecontextFeaturizer
from paper_comparison.metrics import SchemaRecallMetric
import json

gold_file = "data/arxiv_tables_2308_high_quality/test_preds.json"

# tables = [json.loads(x) for x in open(gold_file).readlines()]
data = json.loads(open(gold_file).read())
gold_tables = {}
baseline_gpt_tables = {}
ours_gpt_tables = {}
for table in data["gold"]:
    gold_tables[table["tab_id"]] = table
for table in data["pred"]["hard"]["baseline"]["gpt4"]:
    baseline_gpt_tables[table["tab_id"]] = table
for table in data["pred"]["hard"]["ours"]["gpt4"]:
    ours_gpt_tables[table["tab_id"]] = table

emscorer = ExactMatchScorer()
edscorer = EditDistanceScorer()
jscorer = JaccardAlignmentScorer(remove_stopwords=False)
stscorer = SentenceTransformerAlignmentScorer()
name_feat = BaseFeaturizer("name")
value_feat = ValueFeaturizer("values")
# decontext_feat = DecontextFeaturizer("decontext")

# TODO: Construct metrics for various featurizer, alignment scorer and sim_threshold combinations
name_js_recall = SchemaRecallMetric(featurizer=value_feat, alignment_scorer=jscorer, sim_threshold=0.3)


# TODO: Change this loop to follow the finalized prediction structure
for try_num in range(0, 10):
    cur_try = f'try_{try_num}'
    for tab_id in gold_tables:
        table = gold_tables[tab_id]
        column_names = list(table['table'].keys())
        gold_table_data = Table(table['tab_id'], set(column_names), table['table'])
            
        if cur_try in baseline_gpt_tables[tab_id]['table']:
            for table in baseline_gpt_tables[tab_id]['table'][cur_try]:
                column_names = list(table.keys())
                cur_pred_table_data = Table(tab_id, set(column_names), table)
                name_js_recall.add(cur_pred_table_data, gold_table_data)

    print(name_js_recall.process_scores())
    