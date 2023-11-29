"""Utility functions for computing metrics"""
from typing import Any

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

stopwords = stopwords.words("english")
punctuation = "()[]{},.?!/''\"``"
ps = PorterStemmer()


def get_keywords(sentence: str) -> set[str]:
    """Extract keywords from a sentence.

    Extract keywords from a sentence by lowercasing, tokenizing and stemming words. Punctuation and
    stopwords ar filtered out. Only unique words are returned.

    Args:
        sentence (str): The text to extract keywords from.

    Returns:
        set[str] containing the keywords in the sentence.
    """

    return set(
        [
            ps.stem(word.lower())
            for word in word_tokenize(sentence)
            if word not in stopwords and word not in punctuation
        ]
    )


def jaccard_sentence(sentence_1: str, sentence_2: str, keywords: bool = True) -> float:
    """Calculate the Jaccard similarity between sentences.

    If `keywords` is True, only calculate the similarity using the keywords. If False, then use all of the tokens
    as tokenized by nltk.word_tokenize.

    Args:
        sentence_1 (str): The first sentence.
        sentence_2 (str): The second sentence.
        keywords (bool): Whether to use keywords (True) or all tokens (False). Default is True.

    Returns:
        The Jaccard similarity between the two sentences (float).
    """

    if keywords:
        sentence_1_tokens = get_keywords(sentence_1)
        sentence_2_tokens = get_keywords(sentence_2)
    else:
        sentence_1_tokens = set(word_tokenize(sentence_1))
        sentence_2_tokens = set(word_tokenize(sentence_2))
    return jaccard(sentence_1_tokens, sentence_2_tokens)


def jaccard(a: set[Any], b: set[Any]) -> float:
    """Calculate and return the Jaccard similarity between set `a` and `b`."""

    return len(a & b) / len(a | b)


def get_p_r_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Get the precision, recall, and F1 given true positives, false positives, and false negatives.

    Args:
        tp (int): True positives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        Tuple[float, float, float] containing (precision, recall, F1).
    """

    if tp == 0:
        return 0, 0, 0
    else:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        return p, r, 2 * p * r / (p + r)


def get_similar_sentence(query: str, pool: list[str], method="sentence_transformer") -> tuple[str, float]:
    """
    Returns the most similar string in the pool to the query along with the similarity scores.
    Uses
    """
    if method == "sentence_transformer":
        from sentence_transformers import SentenceTransformer, util

        # See: https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models
        # Fast but worse: all-MiniLM-L6-v2
        # Slow but better: all-mpnet-base-v2
        # not sure how these do with scientific text...
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([query] + pool, convert_tensors=True)
        scores = util.cos_sim(embeddings[0], embeddings[1:]).numpy()
        max_sent_idx = np.argmax(scores)
        return pool[max_sent_idx], scores[max_sent_idx]

    elif method in ("jaccard_keywords", "jaccard"):
        keywords = method == "jaccard_keywords"
        scores = [jaccard_sentence(query, sentence, keywords=keywords) for sentence in pool]
        max_sent_idx = np.argmax(scores)
        return pool[max_sent_idx], scores[max_sent_idx]

    else:
        raise ValueError(f"Invalid similarity method: {method}")
