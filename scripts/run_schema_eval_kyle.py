"""

Run Schema Eval.

This is mostly adapted from Aakanksha's code, and it runs on Aakanksha's value generations,
which basically are an augmentation of Yoonjoo's schema generation files.

All data is on S3:

    aws s3 sync s3://ai2-s2-research-public/lit-review-tables-emnlp-2024/final_data/ final_data/

The data is organized as follows:

|-- final_data/
    |-- baseline/
        |-- metric_validation_0/
    |-- gold/
        |-- metric_validation_0_full_texts/
            |-- dataset_with_ics.jsonl
            ...
    |-- value_preds/
        |-- metric_validation_0_setting2/
        |-- metric_validation_0_setting3/
        |-- metric_validation_0_setting4/
        |-- metric_validation_0_setting5/
    |-- schema_preds/
        ...
    |-- manual_inspection/
        ...

"""

import json
import os
import pickle
import random
import time
from collections import Counter
from glob import glob

import pandas as pd
from tqdm import tqdm

from paper_comparison.metrics import SchemaRecallMetric
from paper_comparison.metrics_utils import (
    BaseFeaturizer,
    DecontextFeaturizer,
    EditDistanceScorer,
    ExactMatchScorer,
    JaccardAlignmentScorer,
    Llama3AlignmentScorer,
    SentenceTransformerAlignmentScorer,
    ValueFeaturizer,
)
from paper_comparison.types import Table

# os.environ["OPENAI_API_KEY"] = "ADD_API_KEY"


assert os.environ["OPENAI_API_KEY"] is not None
assert os.environ["TOGETHER_API_KEY"] is not None

# scoring functions
emscorer = ExactMatchScorer()
edscorer = EditDistanceScorer()
jscorer = JaccardAlignmentScorer(remove_stopwords=False)
jscorer_nostop = JaccardAlignmentScorer(remove_stopwords=True)
stscorer = SentenceTransformerAlignmentScorer()
llama3scorer = Llama3AlignmentScorer()


# featurization functions
name_feat = BaseFeaturizer("name")
value_feat = ValueFeaturizer("values")
decontext_feat = DecontextFeaturizer("decontext")


# mapping
MAPPING = {
    "metric_validation_0__title_abstract__baseline__gpt3.5_mixtral": {
        "value_pred_dir": "/Users/kylel/ai2/paper_comparison_internal/final_data/baseline/metric_validation_0/",
        "gold_file": "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/tables.jsonl",
        "metadata_file": "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/dataset_with_ics.jsonl",
    },
    "metric_validation_0__title_abstract__gen_caption__gpt_mixtral": {
        "value_pred_dir": "/Users/kylel/ai2/paper_comparison_internal/final_data/value_preds/metric_validation_0_setting2/",
        "gold_file": "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/tables.jsonl",
        "metadata_file": "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/dataset_with_ics.jsonl",
    },
    "metric_validation_0__title_abstract__gold_caption__gpt_mixtral": {
        "value_pred_dir": "/Users/kylel/ai2/paper_comparison_internal/final_data/value_preds/metric_validation_0_setting3/",
        "gold_file": "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/tables.jsonl",
        "metadata_file": "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/dataset_with_ics.jsonl",
    },
    "metric_validation_0__title_abstract__caption_inline_refs__gpt_mixtral": {
        "value_pred_dir": "/Users/kylel/ai2/paper_comparison_internal/final_data/value_preds/metric_validation_0_setting4/",
        "gold_file": "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/tables.jsonl",
        "metadata_file": "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/dataset_with_ics.jsonl",
    },
    "metric_validation_0__title_abstract__ics_example__gpt_mixtral": {
        "value_pred_dir": "/Users/kylel/ai2/paper_comparison_internal/final_data/value_preds/metric_validation_0_setting5/",
        "gold_file": "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/tables.jsonl",
        "metadata_file": "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/dataset_with_ics.jsonl",
    },
}


def validate_data_paths():
    for setting, paths in MAPPING.items():
        assert os.path.exists(paths["value_pred_dir"])
        assert os.path.exists(paths["gold_file"])
        assert os.path.exists(paths["metadata_file"])


def dummy_test_metric():
    test_table = Table(
        tabid="test",
        schema={"a", "b", "c"},
        values={"a": ["1", "2", "3"], "b": ["4", "5", "6"], "c": ["7", "8", "9"]},
    )
    metric = SchemaRecallMetric(featurizer=name_feat, alignment_scorer=emscorer, sim_threshold=0.999)
    metric.add(test_table, test_table)
    print(metric.process_scores())
    metric = SchemaRecallMetric(featurizer=decontext_feat, alignment_scorer=stscorer, sim_threshold=0.999)
    metric.add(test_table, test_table)
    print(metric.process_scores())


def dummy_test_alignment_scorer():
    assert emscorer.calculate_pair_similarity(prediction="a", target="a") == 1.0
    assert emscorer.calculate_pair_similarity(prediction="a", target="a b") == 0.0
    assert jscorer.calculate_pair_similarity(prediction="a", target="a") == 1.0
    assert jscorer.calculate_pair_similarity(prediction="a", target="a b") == 0.5
    scores = stscorer.calculate_pair_similarity(predictions=["a jumping cat"], targets=["a hopping kitten"])
    assert scores.tolist()[0][0] > 0.6


def dummy_test_llama3_schema_alignment():
    pred_table_values = {
        "Studies decontextualization?": {"choi21": ["yes"], "newman23": ["yes"], "potluri23": ["no"]},
        "What is their data source?": {
            "choi21": ["Wikipedia"],
            "newman23": ["Scientific Papers"],
            "potluri23": ["ELI5"],
        },
        "What field are they in?": {"choi21": ["NLP"], "newman23": ["NLP"], "potluri23": ["NLP"]},
        "What task do they study?": {
            "choi21": ["decontextualization"],
            "newman23": ["decontextualization"],
            "potluri23": ["long-answer summarization"],
        },
    }

    target_table_values = {
        "Decontext": {"choi21": ["yes"], "newman23": ["yes"], "potluri23": ["no"]},
        "Data": {
            "choi21": ["Wikipedia"],
            "newman23": ["Scientific Papers"],
            "potluri23": ["ELI5"],
        },
        "Field": {"choi21": ["NLP"], "newman23": ["NLP"], "potluri23": ["NLP"]},
        "Task": {
            "choi21": ["decontextualization"],
            "newman23": ["decontextualization"],
            "potluri23": ["long-answer summarization"],
        },
    }

    pred_table = Table(
        tabid="0",
        schema=set(pred_table_values.keys()),
        values=pred_table_values,
    )

    target_table = Table(
        tabid="0",
        schema=set(target_table_values.keys()),
        values=target_table_values,
    )

    # now im gonna try to use llama3, but allowing different variations of featurization
    # OK; i walked through the code, and the full table is displayed regardless of featurization
    alignments = llama3scorer.score_schema_alignments(
        pred_table=pred_table, gold_table=target_table, featurizer=name_feat
    )
    print(alignments)


def dummy_test_featurizer():
    test_table = Table(
        tabid="test",
        schema={"precision", "recall", "f1"},
        values={
            "precision": ["0.5", "0.3", "0.1"],
            "recall": ["0.4", "0.9", "0.4"],
            "f1": ["0.45", "0.45", "0.16"],
        },
    )
    # requires order, since the output of decontext assumes same order
    column_names = ["precision", "recall", "f1"]

    # test value featurizer
    results = value_feat.featurize(column_names=column_names, table=test_table)

    start = time.time()
    results = decontext_feat.featurize(column_names=column_names, table=test_table)
    end = time.time()
    print(f"Time taken (in seconds): {round(end - start, 2)}")

    # organize decontext results
    decontext_schema = {column_name: result for result, column_name in zip(results, column_names)}
    from pprint import pprint

    pprint(decontext_schema)

    # caching
    assert test_table.decontext_schema is None
    test_table.decontext_schema = decontext_schema

    # re-run should use cache now
    start = time.time()
    rerun = decontext_feat.featurize(column_names=column_names, table=test_table)
    end = time.time()
    print(f"Time taken (in seconds): {round(end - start, 2)}")


def get_instance_dicts(pred_dir: str) -> list[dict]:
    """Just a function that traverses paths in a directory"""
    instance_dicts = []
    for instance_id in os.listdir(pred_dir):
        dir0 = os.path.join(pred_dir, instance_id)
        # skips random files
        if not os.path.isdir(dir0):
            continue
        for base_model in os.listdir(dir0):
            # skips random files
            dir1 = os.path.join(dir0, base_model)
            if not os.path.isdir(dir1):
                continue
            for prompt_method in os.listdir(dir1):
                dir2 = os.path.join(dir1, prompt_method)
                # one more time, random files
                if not os.path.isdir(dir2):
                    continue
                for try_num in os.listdir(dir2):
                    instance_dicts.append(
                        {
                            "instance_id": instance_id,
                            "base_model": base_model,
                            "prompt_method": prompt_method,
                            "try_num": try_num,
                            "path": os.path.join(dir2, try_num),
                        }
                    )
    return instance_dicts


def _open_pred_table(file):
    """Opens up table object

    Test:
        table = open_pred_table(file="/Users/kylel/ai2/paper_comparison_internal/final_data/value_preds/metric_validation_0_setting5/cbec0df5-c40f-4cef-92bc-a57de8bf5c0e/gpt3.5/ours_outputs/try_0_with_values.json")
    """
    with open(file, "r") as f:
        table_jsons = json.load(f)
    # we take the first one. this is a List field because we used to do 5 generations per.
    # now only 1, but some old pred files may have multiple.
    table_json = table_jsons[0]
    table_instance = Table(
        tabid=table_json["tabid"], schema=list(table_json["table"].keys()), values=table_json["table"]
    )
    return table_instance


def open_pred_tables(setting: str, base_model: str) -> list:
    pred_dir = MAPPING[setting]["value_pred_dir"]

    instance_dicts = get_instance_dicts(pred_dir)
    instance_dicts = [
        d
        for d in instance_dicts
        if d["try_num"] == "try_0_with_better_values.json" and d["base_model"] == base_model
    ]

    pred_tables = []
    for instance_dict in instance_dicts:
        pred_table = _open_pred_table(file=instance_dict["path"])
        pred_tables.append(pred_table)
    return pred_tables


def open_baseline_tables(setting: str, base_model: str) -> list:
    pred_dir = MAPPING[setting]["value_pred_dir"]
    gold_file = MAPPING[setting]["gold_file"]
    metadata_file = MAPPING[setting]["metadata_file"]

    # load gold data as a lookup dict
    instance_id_to_gold_tables = open_gold_tables(gold_file, metadata_file)

    instance_dicts = get_instance_dicts(pred_dir)
    instance_dicts = [
        d
        for d in instance_dicts
        if d["try_num"] == "try_0.json"
        and d["prompt_method"] == "baseline_outputs"
        and d["base_model"] == base_model
        and d["instance_id"] in instance_id_to_gold_tables
    ]

    pred_tables = []
    for instance_dict in instance_dicts:
        pred_table = _open_pred_table(file=instance_dict["path"])
        pred_tables.append(pred_table)
    return pred_tables


def _open_instance_id_to_table_metadata(file) -> dict:
    """Opens up gold tables

    Test:
        tables = open_instance_id_to_table_metadata(file="/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/dataset_with_ics.jsonl")
    """
    instance_id_to_table_metadata = {}
    with open(file, "r") as f:
        for line in f:
            table_dict = json.loads(line)
            tabid = table_dict["_table_hash"]
            caption = table_dict["caption"]
            inline_ref = table_dict["in_text_ref"]
            instance_id_to_table_metadata[tabid] = {"caption": caption, "inline_ref": inline_ref}
    return instance_id_to_table_metadata


def open_gold_tables(tables_path, metadata_path) -> dict:
    """Opens up gold tables

    Test:
        tables = open_gold_tables(file="/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/tables.jsonl")
    """
    instance_id_to_table_metadata = _open_instance_id_to_table_metadata(metadata_path)

    instance_id_to_table = {}
    with open(tables_path, "r") as f:
        for line in f:
            table_dict = json.loads(line)
            tabid = table_dict["tabid"]
            metadata_dict = instance_id_to_table_metadata[tabid]
            table = Table(
                tabid=tabid,
                schema=list(table_dict["table"].keys()),
                values=table_dict["table"],
                caption=metadata_dict["caption"],
            )
            table.inline_ref = metadata_dict["inline_ref"]
            instance_id_to_table[table.tabid] = table
    return instance_id_to_table


def table_to_dataframe(table: Table) -> pd.DataFrame:
    """Converts a Table object to a pandas DataFrame"""
    df = pd.DataFrame(data=table.values)
    return df


def metric_scores_to_json(scores: dict, outfile: str):
    tabid_to_recall_scores = {tabid: scoress[0] for tabid, scoress in scores["recall"].items()}
    tabid_to_alignments = {
        tabid: [{"gold": gold, "pred": pred} for (gold, pred), score in alignmentss[0].items()]
        for tabid, alignmentss in scores["alignments"].items()
    }
    with open(outfile, "w") as f:
        json.dump({"recall": tabid_to_recall_scores, "alignments": tabid_to_alignments}, f, indent=4)


if __name__ == "__main__":
    # sample some files to visualize
    OUTDIR = "/Users/kylel/ai2/paper_comparison_internal/final_data/manual_inspection/"
    os.makedirs(OUTDIR, exist_ok=True)
    for setting in MAPPING.keys():
        for base_model in ["gpt3.5", "mixtral"]:
            outfile = os.path.join(OUTDIR, f"{setting}__{base_model}.txt")

            gold_file = MAPPING[setting]["gold_file"]
            metadata_file = MAPPING[setting]["metadata_file"]
            instance_id_to_gold_tables = open_gold_tables(gold_file, metadata_file)

            if "baseline" in setting:
                pred_tables = open_baseline_tables(setting=setting, base_model=base_model)
            else:
                pred_tables = open_pred_tables(setting=setting, base_model=base_model)

            random.seed(42)
            pred_tables = random.sample(pred_tables, 10)

            with open(outfile, "w") as f:
                for pred_table in tqdm(pred_tables):
                    gold_table = instance_id_to_gold_tables[pred_table.tabid]
                    f.write(
                        f"======================================={pred_table.tabid}============================================\n"
                    )
                    f.write(f"{table_to_dataframe(gold_table).to_markdown(index=False)}\n\n")
                    f.write(f"Table X: {gold_table.caption}\n\n")
                    f.write(
                        f"------------------------------------------------------------------------------------------------\n\n"
                    )
                    f.write(f"{table_to_dataframe(pred_table).to_markdown(index=False)}\n")
                    f.write(
                        f"------------------------------------------------------------------------------------------------\n\n"
                    )

                    alignments = llama3scorer.score_schema_alignments(pred_table=pred_table, gold_table=gold_table)
                    for (gold_column, pred_column), score in alignments.items():
                        if score == 1.0:
                            f.write(f"{pred_column} <--> {gold_column}\n")
                    f.write(
                        f"================================================================================================\n\n\n\n"
                    )

    # count N/A in pred tables
    setting_metric_to_nas = Counter()
    setting_metric_to_slots = Counter()
    for setting in MAPPING.keys():
        for base_model in ["gpt3.5", "mixtral"]:
            if "baseline" in setting:
                pred_tables = open_baseline_tables(setting=setting, base_model=base_model)
            else:
                pred_tables = open_pred_tables(setting=setting, base_model=base_model)
            for pred_table in pred_tables:
                for column in pred_table.schema:
                    num_na = sum([value == "N/A" for paper_id, value in pred_table.values[column].items()])
                    setting_metric_to_nas[f"{setting}__{base_model}"] += num_na
                    setting_metric_to_slots[f"{setting}__{base_model}"] += len(pred_table.values[column])
    with open("/Users/kylel/ai2/paper_comparison_internal/final_data/manual_inspection/nas.txt", "w") as f:
        f.write("N/A counts\n\n")
        json.dump(dict(setting_metric_to_nas), f, indent=4)
        f.write("\n\n")
        f.write("Slot counts\n\n")
        json.dump(dict(setting_metric_to_slots), f, indent=4)
        f.write("\n\n")
        for k, v in setting_metric_to_nas.items():
            f.write(f"{k}: {v} / {setting_metric_to_slots[k]} = {round(v / setting_metric_to_slots[k], 2)}\n")

    # jaccard metric calculation
    with open(f"/Users/kylel/ai2/paper_comparison_internal/final_data/manual_inspection/metrics.txt", "a") as f:
        for setting in MAPPING.keys():
            gold_file = MAPPING[setting]["gold_file"]
            metadata_file = MAPPING[setting]["metadata_file"]
            instance_id_to_gold_tables = open_gold_tables(gold_file, metadata_file)
            for base_model in ["gpt3.5", "mixtral"]:
                if "baseline" in setting:
                    pred_tables = open_baseline_tables(setting=setting, base_model=base_model)
                else:
                    pred_tables = open_pred_tables(setting=setting, base_model=base_model)
                for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    for scorer in [emscorer, jscorer_nostop]:
                        for featurizer in [
                            # name_feat,
                            value_feat
                        ]:
                            METRIC = SchemaRecallMetric(
                                featurizer=featurizer, alignment_scorer=scorer, sim_threshold=threshold
                            )
                            for pred_table in tqdm(pred_tables):
                                gold_table = instance_id_to_gold_tables[pred_table.tabid]
                                METRIC.add(pred_table, gold_table)
                            # now export it all
                            scores_dict = METRIC.process_scores()
                            f.write(
                                f"config={setting}__model={base_model}__threshold={threshold}__scorer={scorer.name}__featurizer={featurizer.name}\n"
                            )
                            json.dump(scores_dict, f)
                            f.write("\n\n")

    # metric calculation, but with llama3 + outputting alignments too
    with open(
        f"/Users/kylel/ai2/paper_comparison_internal/final_data/manual_inspection/metrics_llama3__gpt35.txt", "w"
    ) as f:
        for setting in MAPPING.keys():
            gold_file = MAPPING[setting]["gold_file"]
            metadata_file = MAPPING[setting]["metadata_file"]
            instance_id_to_gold_tables = open_gold_tables(gold_file, metadata_file)
            for base_model in [
                "gpt3.5"
                #    , "mixtral"
            ]:
                if "baseline" in setting:
                    pred_tables = open_baseline_tables(setting=setting, base_model=base_model)
                else:
                    pred_tables = open_pred_tables(setting=setting, base_model=base_model)
            random.seed(42)
            pred_tables = random.sample(pred_tables, 100)
            METRIC = SchemaRecallMetric(featurizer=name_feat, alignment_scorer=llama3scorer)
            for pred_table in tqdm(pred_tables):
                gold_table = instance_id_to_gold_tables[pred_table.tabid]
                try:
                    METRIC.add(pred_table, gold_table)
                except Exception as e:
                    continue
            # now export it all
            scores_dict = METRIC.process_scores()
            f.write(f"config={setting}__model={base_model}\n")
            json.dump(scores_dict, f)
            f.write("\n\n")
            # also export alignments
            metric_scores_to_json(
                METRIC.scores,
                f"/Users/kylel/ai2/paper_comparison_internal/final_data/manual_inspection/alignments__llama3__{setting}__{base_model}.json",
            )

    # precomputing decontext information on predicted tables
    CACHE_DECONTEXT_DIR = "/Users/kylel/ai2/paper_comparison_internal/final_data/decontext/"
    os.makedirs(CACHE_DECONTEXT_DIR, exist_ok=True)
    decontext_feat = DecontextFeaturizer("decontext", model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    for setting in MAPPING.keys():
        gold_file = MAPPING[setting]["gold_file"]
        metadata_file = MAPPING[setting]["metadata_file"]
        instance_id_to_gold_tables = open_gold_tables(gold_file, metadata_file)
        for base_model in ["gpt3.5", "mixtral"]:
            if "baseline" in setting:
                pred_tables = open_baseline_tables(setting=setting, base_model=base_model)
            else:
                pred_tables = open_pred_tables(setting=setting, base_model=base_model)
            for pred_table in tqdm(pred_tables):
                column_names = list(pred_table.values.keys())
                features = decontext_feat.featurize(column_names=column_names, table=pred_table)
                column_name_to_feature = {
                    column_name: feature for column_name, feature in zip(column_names, features)
                }
                table_id_to_decontext = {pred_table.tabid: column_name_to_feature}
                outfile = os.path.join(CACHE_DECONTEXT_DIR, f"{setting}__{base_model}__{pred_table.tabid}.json")
                with open(outfile, "w") as f:
                    json.dump(table_id_to_decontext, f, indent=4)

    # precomputing decontext information on gold tables
    GOLD_CACHE_DECONTEXT_DIR = "/Users/kylel/ai2/paper_comparison_internal/final_data/decontext_gold/"
    os.makedirs(GOLD_CACHE_DECONTEXT_DIR, exist_ok=True)
    decontext_feat = DecontextFeaturizer("decontext", model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    gold_file = (
        "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/tables.jsonl"
    )
    metadata_file = "/Users/kylel/ai2/paper_comparison_internal/final_data/gold/metric_validation_0_full_texts/dataset_with_ics.jsonl"
    instance_id_to_gold_tables = open_gold_tables(gold_file, metadata_file)
    for instance_id, gold_table in tqdm(instance_id_to_gold_tables.items()):
        column_names = list(gold_table.values.keys())
        features = decontext_feat.featurize(column_names=column_names, table=gold_table)
        column_name_to_feature = {column_name: feature for column_name, feature in zip(column_names, features)}
        gold_table_id_to_decontext = {gold_table.tabid: column_name_to_feature}
        outfile = os.path.join(GOLD_CACHE_DECONTEXT_DIR, f"{gold_table.tabid}.json")
        with open(outfile, "w") as f:
            json.dump(gold_table_id_to_decontext, f, indent=4)

    # jaccard metric, but with decontext now
    with open(
        f"/Users/kylel/ai2/paper_comparison_internal/final_data/manual_inspection/metrics_decontext.txt", "w"
    ) as f:
        for setting in MAPPING.keys():
            gold_file = MAPPING[setting]["gold_file"]
            metadata_file = MAPPING[setting]["metadata_file"]
            instance_id_to_gold_tables = open_gold_tables(gold_file, metadata_file)
            for base_model in ["gpt3.5", "mixtral"]:
                if "baseline" in setting:
                    pred_tables = open_baseline_tables(setting=setting, base_model=base_model)
                else:
                    pred_tables = open_pred_tables(setting=setting, base_model=base_model)
                for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    for scorer in [jscorer_nostop]:
                        for featurizer in [decontext_feat]:
                            METRIC = SchemaRecallMetric(
                                featurizer=featurizer, alignment_scorer=scorer, sim_threshold=threshold
                            )
                            for pred_table in tqdm(pred_tables):
                                gold_table = instance_id_to_gold_tables[pred_table.tabid]
                                # load decontext information for gold
                                gold_decontext_file = os.path.join(
                                    GOLD_CACHE_DECONTEXT_DIR, f"{pred_table.tabid}.json"
                                )
                                with open(gold_decontext_file, "r") as f_g:
                                    gold_decontext = json.load(f_g)
                                    gold_table.decontext_schema = gold_decontext[gold_table.tabid]
                                # load decontext information for pred
                                pred_decontext_file = os.path.join(
                                    CACHE_DECONTEXT_DIR, f"{setting}__{base_model}__{pred_table.tabid}.json"
                                )
                                with open(pred_decontext_file, "r") as f_p:
                                    pred_decontext = json.load(f_p)
                                    pred_table.decontext_schema = pred_decontext[pred_table.tabid]
                                METRIC.add(pred_table, gold_table)
                            # now export it all
                            scores_dict = METRIC.process_scores()
                            f.write(
                                f"config={setting}__model={base_model}__threshold={threshold}__scorer={scorer.name}__featurizer={featurizer.name}\n"
                            )
                            json.dump(scores_dict, f)
                            f.write("\n\n")

    # finally, llama3 + outputting alignments too + decontext cols
    with open(
        f"/Users/kylel/ai2/paper_comparison_internal/final_data/manual_inspection/metrics_llama3_decontext__gpt35.txt",
        "w",
    ) as f:
        for setting in MAPPING.keys():
            gold_file = MAPPING[setting]["gold_file"]
            metadata_file = MAPPING[setting]["metadata_file"]
            instance_id_to_gold_tables = open_gold_tables(gold_file, metadata_file)
            for base_model in [
                "gpt3.5"
                # , "mixtral"
            ]:
                if "baseline" in setting:
                    pred_tables = open_baseline_tables(setting=setting, base_model=base_model)
                else:
                    pred_tables = open_pred_tables(setting=setting, base_model=base_model)
            random.seed(42)
            pred_tables = random.sample(pred_tables, 100)
            METRIC = SchemaRecallMetric(featurizer=decontext_feat, alignment_scorer=llama3scorer)
            for pred_table in tqdm(pred_tables):
                gold_table = instance_id_to_gold_tables[pred_table.tabid]
                # load decontext information for gold
                gold_decontext_file = os.path.join(GOLD_CACHE_DECONTEXT_DIR, f"{pred_table.tabid}.json")
                with open(gold_decontext_file, "r") as f_g:
                    gold_decontext = json.load(f_g)
                    gold_table.decontext_schema = gold_decontext[gold_table.tabid]
                # load decontext information for pred
                pred_decontext_file = os.path.join(
                    CACHE_DECONTEXT_DIR, f"{setting}__{base_model}__{pred_table.tabid}.json"
                )
                with open(pred_decontext_file, "r") as f_p:
                    pred_decontext = json.load(f_p)
                    pred_table.decontext_schema = pred_decontext[pred_table.tabid]
                try:
                    METRIC.add(pred_table, gold_table)
                except Exception as e:
                    continue
            # now export it all
            scores_dict = METRIC.process_scores()
            f.write(f"config={setting}__model={base_model}__decontext\n")
            json.dump(scores_dict, f)
            f.write("\n\n")
            # also export alignments
            metric_scores_to_json(
                METRIC.scores,
                f"/Users/kylel/ai2/paper_comparison_internal/final_data/manual_inspection/alignments__llama3__{setting}__{base_model}__decontext.json",
            )

    # SBERT matching, but with decontext + also output alignments
    with open(
        f"/Users/kylel/ai2/paper_comparison_internal/final_data/manual_inspection/metrics_decontext_sbert.txt", "w"
    ) as f:
        for setting in MAPPING.keys():
            gold_file = MAPPING[setting]["gold_file"]
            metadata_file = MAPPING[setting]["metadata_file"]
            instance_id_to_gold_tables = open_gold_tables(gold_file, metadata_file)
            for base_model in ["gpt3.5", "mixtral"]:
                if "baseline" in setting:
                    pred_tables = open_baseline_tables(setting=setting, base_model=base_model)
                else:
                    pred_tables = open_pred_tables(setting=setting, base_model=base_model)
                for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    for scorer in [stscorer]:
                        for featurizer in [decontext_feat]:
                            METRIC = SchemaRecallMetric(
                                featurizer=featurizer, alignment_scorer=scorer, sim_threshold=threshold
                            )
                            for pred_table in tqdm(pred_tables):
                                gold_table = instance_id_to_gold_tables[pred_table.tabid]
                                # load decontext information for gold
                                gold_decontext_file = os.path.join(
                                    GOLD_CACHE_DECONTEXT_DIR, f"{pred_table.tabid}.json"
                                )
                                with open(gold_decontext_file, "r") as f_g:
                                    gold_decontext = json.load(f_g)
                                    gold_table.decontext_schema = gold_decontext[gold_table.tabid]
                                # load decontext information for pred
                                pred_decontext_file = os.path.join(
                                    CACHE_DECONTEXT_DIR, f"{setting}__{base_model}__{pred_table.tabid}.json"
                                )
                                with open(pred_decontext_file, "r") as f_p:
                                    pred_decontext = json.load(f_p)
                                    pred_table.decontext_schema = pred_decontext[pred_table.tabid]
                                METRIC.add(pred_table, gold_table)
                            # now export it all
                            scores_dict = METRIC.process_scores()
                            f.write(
                                f"config={setting}__model={base_model}__threshold={threshold}__scorer={scorer.name}__featurizer={featurizer.name}\n"
                            )
                            json.dump(scores_dict, f)
                            f.write("\n\n")
                            # also export alignments
                            metric_scores_to_json(
                                METRIC.scores,
                                f"/Users/kylel/ai2/paper_comparison_internal/final_data/manual_inspection/alignments__{setting}__{base_model}__{scorer.name}__{featurizer.name}.json",
                            )
