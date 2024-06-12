import os
import json
import pickle
from pathlib import Path

os.environ["OPENAI_API_KEY"] = "ADD_API_KEY"
os.environ["TOGETHER_API_KEY"] = "ADD_API_KEY"

from paper_comparison.types import Table
from paper_comparison.metrics_utils import JaccardAlignmentScorer, ExactMatchScorer, EditDistanceScorer, SentenceTransformerAlignmentScorer, Llama3AlignmentScorer
from paper_comparison.metrics_utils import BaseFeaturizer, ValueFeaturizer, DecontextFeaturizer
from paper_comparison.metrics import SchemaRecallMetric

# Change subset name to eval on different splits
subset = '../../medium/'
model_names = ['gpt3.5', 'mixtral']
variant_names = ['ours']
# threshold_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
threshold_values = [0.5]

emscorer = ExactMatchScorer()
edscorer = EditDistanceScorer()
jscorer = JaccardAlignmentScorer(remove_stopwords=False)
jscorer_nostop = JaccardAlignmentScorer(remove_stopwords=True)
stscorer = SentenceTransformerAlignmentScorer()
llama_scorer = Llama3AlignmentScorer("llama", debug=True)
name_feat = BaseFeaturizer("name")
value_feat = ValueFeaturizer("values")
decontext_feat = DecontextFeaturizer("decontext")

featurizers = [name_feat, value_feat, decontext_feat]
# scorers = [jscorer_nostop, emscorer, edscorer]
# scorers = [stscorer]
scorers = [llama_scorer]


with open("../data/v4/metric_validation_0_full_texts/tables.jsonl") as f:
    gold_tables = [json.loads(line) for line in f]

pred_tables_caption_itref = []
for sample in gold_tables:
    try:
        with open(f"../../generations/metric_validation_0_setting4_caption_itref/{sample['tabid']}/gpt3.5/ours_outputs/try_0_with_values.json") as f:
            pred_tables_caption_itref.append(json.load(f)[-1])
    except FileNotFoundError:
        pred_tables_caption_itref.append(None)


metric = SchemaRecallMetric(featurizer=featurizers[0], alignment_scorer=scorers[0], sim_threshold=0.5)
for sample_idx in range(16, 20):
    print(sample_idx)
    gold_table_input = gold_tables[sample_idx]
    gold_table_instance = Table(gold_table_input["tabid"], list(gold_table_input["table"].keys()), gold_table_input["table"])

    pred_table_input = pred_tables_caption_itref[sample_idx]
    pred_table_instance = Table(pred_table_input["tabid"], list(pred_table_input["table"].keys()), pred_table_input["table"])
    metric.add(pred_table_instance, gold_table_instance)

print(metric.process_scores())