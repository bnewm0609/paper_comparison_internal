import pytest

from paper_comparison import metrics
from paper_comparison.types import Table


def test_SchemaPrecisionRecallMetric():
    prf_metric = metrics.SchemaPrecisionRecallMetric(args=None)

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

    prf_metric.add(
        prediction=pred_table,
        target=target_table,
    )
    scores = prf_metric.process_scores()
    assert scores["precision"] == 1.0 and scores["recall"] == 1.0 and scores["f1"] == 1.0
    prf_metric.reset()

    del pred_table_values["What field are they in?"]
    pred_table = Table(
        tabid="0",
        schema=set(pred_table_values.keys()),
        values=pred_table_values,
    )

    prf_metric.add(
        prediction=pred_table,
        target=target_table,
    )
    scores = prf_metric.process_scores()
    assert scores["precision"] == 1.0 and scores["recall"] == 0.75
    prf_metric.reset()

    pred_table_values["What sub field do they study?"] = {
        "choi21": ["NLP"],
        "newman23": ["NLP"],
        "potluri23": ["NLP"],
    }
    pred_table = Table(
        tabid="0",
        schema=set(pred_table_values.keys()),
        values=pred_table_values,
    )

    prf_metric = metrics.SchemaPrecisionRecallMetric(args=None, sim_threshold=0.45)
    prf_metric.add(
        prediction=pred_table,
        target=target_table,
    )
    scores = prf_metric.process_scores()
    assert scores["precision"] == 1.0 and scores["recall"] == 1.0


def test_SchemaDiversityMetric():
    diversity_metric = metrics.SchemaDiversityMetric(None)

    pred_schema = {
        "Studies decontextualization?",
        "What field are they in?",
        "What is their data source?",
        "What task do they study?",
    }
    pred_table = Table(tabid="0", schema=pred_schema, values={})
    diversity_metric.add(prediction=pred_table, target=None)
    scores = diversity_metric.process_scores()
    assert scores["diversity"]["self-bleu"] == pytest.approx(10.284844530787057)

    pred_schema = {
        "What field are they in?",
        "What field do they study?",
        "What scientific field do they study?",
    }
    pred_table = Table(tabid="0", schema=pred_schema, values={})
    diversity_metric.add(prediction=pred_table, target=None)
    scores = diversity_metric.process_scores()
    assert scores["diversity"]["self-bleu"] == pytest.approx(33.09915554810221)
