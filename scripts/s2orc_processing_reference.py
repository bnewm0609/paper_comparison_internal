"""
This file holds a number of functions related to processing s2orc data for extracting citation,
section information, etc.
"""

import json
import re

from nltk.tokenize import sent_tokenize
# import spacy
# import scispacy

WINDOW_SIZE = 1500
SENTENCE_WINDOW = 3

# nlp = spacy.load("en_core_sci_lg", disable=["ner"])

def get_sorted_filtered_annotations(annotations):
    annotations = json.loads(annotations)
    
    # remove repeated annotations
    seen_starts = set()
    annotations_ = []
    for annotation in annotations:
        if annotation['start'] in seen_starts:
            continue
        annotations_.append(annotation)
        seen_starts.add(annotation['start'])
    annotations = annotations_
#     annotations = set(annotations)

    # sort the annotations by start position:
    sorted_annotations = sorted(annotations, key=lambda x: int(x['start']))
    return sorted_annotations


def fuzzy_substring(needle, haystack):
    """Calculates the fuzzy match of needle in haystack,
    using a modified version of the Levenshtein distance
    algorithm.
    The function is modified from the levenshtein function
    in the bktree module by Adam Hupp
    
    Reference: 
    https://web.archive.org/web/20200809192640/http://ginstrom.com/scribbles/2007/\
        12/01/fuzzy-substring-matching-with-levenshtein-distance-in-python/
    
    """
    m, n = len(needle), len(haystack)

    # base cases
    if m == 1:
        return not needle in haystack
    if not n:
        return m

    row1 = [0] * (n+1)
    for i in range(0,m):
        row2 = [i+1]
        for j in range(0,n):
            cost = ( needle[i] != haystack[j] )

            row2.append( min(row1[j+1]+1, # deletion
                               row2[j]+1, #insertion
                               row1[j]+cost) #substitution
                           )
        row1 = row2
    return min(row1)

def get_intro_2(s2orc_parse, verbose=False):
    """
    This one is a bit better than the one below in that if it can't find a section called intro
    it looks for the text following the abstract, which sometimes helps. The cost is that it's
    much slower because it has to call a fuzzy string matching function.
    """
    grobid_parse = s2orc_parse['content']['grobid']
    if not grobid_parse:
        return ""

    annotations = grobid_parse.get("annotations", {}).get("section_header", None)
    full_text = grobid_parse.get("contents", "")
    
    if not annotations or not full_text:
        return ""

    sorted_annotations = get_sorted_filtered_annotations(annotations)
    for i, annotation in enumerate(sorted_annotations):
        try:
            header = full_text[int(annotation['start']): int(annotation['end'])].lower()
        except TypeError:
            print(annotation)
            raise TypeError("slice indices have to be ints or whatever")

        if "intro" in header or ("background" in header and "summary" in header):
            text_start = int(annotation['end'])
            try:
                text_end = int(sorted_annotations[i + 1]['start'])
            except IndexError:
                text_end = len(full_text)
                print("intro is last thing in list", sorted_annotations)
            # print(intro_start, intro_end)
            return full_text[text_start: text_end]
    else:
        # we are in a strange situation where the intro is not marked. In this situation,
        # we assume that whatever is between the end of the abstract and the beginning of
        # the section following the abstract is the intro.
        abstract = s2orc_parse['metadata']['abstract']
        try:
            abstract_idx = full_text.index(abstract)
        except ValueError:
            abstract_idx = -1

        if abstract_idx < 0:
            # the abstract doesn't exist verbatim in the full text. This could be because the ArXiv abstract
            # differs from what's in the full text or a pdf processing issue. In these cases, use the 
            # most similar 

            # it's possible that splitting on a single line would also work, but that might split up the abstract too, so let's
            # start with splitting on two newlines.
            paragraphs = full_text.split("\n\n")
            closest_distance = float("inf")
            closest_distance_idx = -1
            print("> Running fuzzy substring search... this might take a ~1min!")
            for para_i, para in enumerate(paragraphs):
                # ed = edit_distance(para, abstract)#, max_ed=closest_distance)  # not using max_ed right now because if it's over max, returns 0 which is confusing
                ed = fuzzy_substring(abstract, para)  # this is quite slow
                # if ed == 0 and abstract != para:
                #     continue
                if ed < closest_distance:
                    # print(ed)
                    closest_distance = ed
                    closest_distance_idx = para_i
                if ed < 10:
                    # exit early if we're "close enough"
                    break
            print("> Done with fuzzy substring search!\n")

            if verbose:
                print(abstract)
                print("closest_distance idx:", closest_distance_idx, "\tdistance:", closest_distance)
                print("Closest paragraph:", paragraphs[closest_distance_idx])
                print()
            # reassign abstract to be the abstract *in the full text* if the edit distance is close enough
            if closest_distance < 305:  # arbitrary constant
                abstract = paragraphs[closest_distance_idx]
                if verbose:
                    print("Reassigned")
                # print("\n", "Reassigned to:\n", abstract)
                abstract_idx = full_text.index(abstract)

        if abstract_idx >= 0:  # we found the abstract
            intro_start = abstract_idx + len(abstract)
            # the intro ends at the first marked section after the end of the abstract
            intro_end = intro_start
            for annotation_i, annotation in enumerate(sorted_annotations):
                # Use another arbitrary constant: The intro should be early on. If it's too late there was
                # probably some issue
                if annotation_i == 2:
                    return ""
                if int(annotation["start"]) > intro_start:
                    intro_end = int(annotation["start"])
                    break
            # print(intro_start, intro_end)
            return full_text[intro_start: intro_end].strip()

    return ""



def get_intro(grobid_parse):
    if not grobid_parse:
        return ""

    annotations = grobid_parse.get("annotations", {}).get("section_header", None)
    full_text = grobid_parse.get("contents", "")
    
    if not annotations or not full_text:
        return ""

    sorted_annotations = get_sorted_filtered_annotations(annotations)
    for i, annotation in enumerate(sorted_annotations):
        try:
            header = full_text[int(annotation['start']): int(annotation['end'])].lower()
        except TypeError:
            print(annotation)
            raise TypeError("slice indices have to be ints or whatever")
        if "intro" in header:
            text_start = int(annotation['end'])
            try:
                text_end = int(sorted_annotations[i + 1]['start'])
            except IndexError:
                text_end = len(full_text)
                print("intro is last thing in list", sorted_annotations)
                # print(header, sorted_annotations)
                # raise IndexError("list index out of range")
#             print(header)
#             print(text_start, text_end)
            return full_text[text_start: text_end]
    else:
        return ""

def get_conclusion(grobid_parse):
    if not grobid_parse:
        return ""

    annotations = grobid_parse.get("annotations", {}).get("section_header", None)
    full_text = grobid_parse.get("contents", "")
    
    if not annotations or not full_text:
        return ""

    sorted_annotations = get_sorted_filtered_annotations(annotations)
    conclusion = ""
    for i, annotation in enumerate(sorted_annotations):
        header = full_text[int(annotation['start']): int(annotation['end'])].lower()
#         print(header)
        if "conclusion" in header or "final remarks" in header:
            text_start = int(annotation['end'])
            if i == len(sorted_annotations) - 1:
                text_end = len(full_text)
            else:
                text_end = int(sorted_annotations[i + 1]['start'])
                
            # don't early exit here by returning because we want the *last* conclusion
            conclusion = full_text[text_start: text_end]
    
    return conclusion

def get_section_ranges(sample):
    """
    The section ranges are represented as a dict with a start, end, and label
    keys, but there's some smarter way to represent them with BSTs but I don't
    want to implement it right now...

    Returns:
        ranges (List<Dict<"start": int, "end": int, "label": str>>):
    """
    sections = json.loads(sample['content']['grobid']['annotations']['section_header'])

    ranges = [{"label": "<bod>", "start": 0}]
    for section in sections:
        start, end = section["start"], section["end"]
        start, end = int(start), int(end)
        label = sample['content']['grobid']['contents'][start: end]
        # let's clean the section labels a bit:
        label = label.strip()\
                    .lower()\
                    .removesuffix(":")\
                    .removesuffix(".")\
                    .strip()

        # let's also not count figures and tables as section headers:
        if label.startswith("fig") or label.startswith("tab"):
            continue

        ranges[-1]["end"] = start
        ranges.append({"label": label, "start": end})

    # close last range with end of document
    ranges[-1]["end"] = len(sample['content']['grobid']['contents'])
    return ranges

def get_section(section_ranges, cite_idx):
    """
    Given section ranges and citation position, returns the

    Args:
        section_ranges (Dict<str, int>): map that's functioning as a named tuple
                                         includes section name, start position,
                                         end position and label for each section
        cite_idx (int): character index of start of citation in the full text

    Returns:
        str: section label
    """
    for r in section_ranges:
        if r["start"] < cite_idx < r["end"]:
            return r["label"]

def get_sentence(span, citation_idx, sentence_window=1, sentences=None):
    """
    Extracts the sentence containing a citation from the span along with surrounding sentences for context.

    Args:
        span (str | List<str>): the span containing the citation OR list of sentences (eg calculated with spacy)
        citation_idx (int): the index of the beginning of the citation in the span
        sentence_window (int): the number of sentences on each side of the citing sentence to return

    Returns:
        Dictionary containing the following key-value pairs
            sentence (str): the sentence with the citation
            sentence_start_idx (int): the character index of the beginning of the sentence
            context (str): the citing sentence with `sentence_window` sentences on either side.
            context_start_idx (int): the character index of the beginning of the context
            List<str>: a list containing `sentence_window` sentences that proceed and follow the sentence with the
                    citation
    """
    if sentences is None:
        # sentences = list(map(str, nlp(span).sents))
        try:
            sentences = sent_tokenize(span)
        except TypeError:
            sentences = list(span)
            print(span)
            print(sentences)
            # raise TypeError("Expected String or Bytes-like object")
    sentence_start_idxs = []
    cumm_length = 0

    result = {
        "sentence": None,
        "sentence_start_idx": None,
        "context": None,
        "context_start_idx": None,
    }

    citing_sent_i = -1

    for sent_i, sentence in enumerate(sentences):

        if cumm_length < citation_idx < cumm_length + len(sentence):
            citing_sent_i = sent_i

        offset = 0
        while not span[cumm_length + offset : cumm_length + len(sentence) + offset] == sentence:
            # Warn if the offset gets too high.
            if offset == 100:  # arbitrary number
                print("possibly in infinite loop...")
                print(f"target: '{sentence}'")
                print(span[cumm_length: cumm_length + len(sentence) + offset])
            offset += 1

        sentence_start_idxs.append(cumm_length + offset)
        cumm_length += len(sentence) + offset
        # sentence_start_idxs.append(cumm_length)
    sentence_start_idxs.append(len(span)) # add final fence-post

    # print(sentence_start_idxs)
    result["sentence"] = sentences[citing_sent_i]
    result["sentence_start_idx"] = sentence_start_idxs[citing_sent_i]
    context_start_idx = sentence_start_idxs[max(0, citing_sent_i - sentence_window)]
    context_end_idx = sentence_start_idxs[min(citing_sent_i + sentence_window + 1, len(sentences))]
    result["context"] = span[context_start_idx: context_end_idx]
    result["context_start_idx"] = context_start_idx

    return result

    #     if citation_idx < cumm_length + len(sentence):
    #         offset = 0
    #         # print(f"target: '{sentence}'")
    #         while not span[cumm_length: cumm_length + len(sentence) + offset].endswith(sentence):
    #             offset += 1
    #         return sentence, cumm_length + offset, sentences[max(0, sent_i - sentence_window): sent_i + sentence_window + 1]
    #     else:
    #         # sent_tokenize does not maintain spaces or new lines between sentences, so cumm_length will be incorrect
    #         # if we just add the length of the extracted sentence. We need to add a small offset to ensure that the
    #         # index of the end of sentence matches up with the actual end of the sentence. This is needed to return the
    #         # character index of the beginning of the citing sentence.
    #         offset = 0
    #         while not span[cumm_length: cumm_length + len(sentence) + offset].endswith(sentence):
    #             # Warn if the offset gets too high.
    #             if offset > 10:
    #                 if offset == 10:
    #                     print("possibly in infinite loop... ")
    #                     print(f"target: '{sentence}'")
    #                     print(span[cumm_length: cumm_length + len(sentence) + offset])
    #             offset += 1

    #         cumm_length += len(sentence) + offset
    #         sentence_start_idxs.append(cumm_length)
    # return None

def get_cited_papers_and_spans(s2orc_dataset, paper_idx, verbose=True):
    bib_refs = json.loads(s2orc_dataset[paper_idx]['content']['grobid']['annotations']['bib_ref'])
    bib_entries = json.loads(s2orc_dataset[paper_idx]['content']['grobid']['annotations']['bib_entry'])

    # we want to map from the ref_id in the bib_ref to the associated bib_entry, so we need to construct
    # a map that's indexed by the ref_id
    bib_entry_map = {}
    for entry in bib_entries:
        v = {}
        if "matched_paper_id" in entry["attributes"]:
            v["matched_paper_id"] = entry["attributes"]["matched_paper_id"]
        assert "doi" not in v
        bib_entry_map[entry["attributes"]["id"]] = v

    if verbose:
        print("== ", s2orc_dataset[paper_idx]['metadata']['title'].strip(), " ==")
    result = []
    window = WINDOW_SIZE
    non_standard_span_window_sizes = {}
    citing_spans = []
    section_ranges = get_section_ranges(s2orc_dataset[paper_idx])
    for header_info in bib_refs:
        try:
            start, end, ref = header_info["start"], header_info["end"], header_info["attributes"]["ref_id"]
            start, end = int(start), int(end)
        except KeyError:
            # KeyError probably bc there's no "attributes", in which case we can't do the linking
            # and should skip this reference
            continue

        section = get_section(section_ranges, start)
        cited_paper = {k: v for k, v in bib_entry_map[ref].items()}
        if not cited_paper:
            # If there's no s2 linked to the paper, skip it. We need the citations to be linked to other papers in the
            # S2 corpus
            continue
        # keep window = 500 here so end + window is consistent
        # but if start < window, update window afterward so the index of the
        # citation is correct when we extract the sentence.
        citing_span = s2orc_dataset[paper_idx]['content']['grobid']['contents'][
            max(0, start - window) : end + window]
        if start < window:
            window = start
            non_standard_span_window_sizes[len(result)] = window

        citing_spans.append(citing_span)
        # citing_sentence_info = get_sentence(citing_span, window, sentence_window=SENTENCE_WINDOW)
        cited_paper['in_text'] = {
            # "start" : window - citing_sentence_info["sentence_start_idx"],
            # "end"   : window + (end - start) - citing_sentence_info["sentence_start_idx"],
            # "context_start": window - citing_sentence_info["context_start_idx"],
            # "context_end": window + (end - start) - citing_sentence_info["context_start_idx"],
            "citing_span": citing_span,
            "span_window": window,
            "start": start,
            "end": end,
            "context_window": SENTENCE_WINDOW,
            "ref_id": ref,
        }

        result.append({
            "section": section,
            # "sentence": citing_sentence_info["sentence"],
            "paper": cited_paper,
            # "context": citing_sentence_info["context"]
        })
        if verbose:
            print({"section": section, "sentence": citing_sentence_info["sentence"], "paper": cited_paper})

    return result

def get_cited_papers_and_sentences(s2orc_dataset, paper_idx, verbose=True):
    result = get_cited_papers_and_spans(s2orc_dataset, paper_idx, verbose)
    # batch the calls to spacy together
    citing_spans = []
    for s in result:
        citing_spans.append(s["paper"]["in_text"].pop("citing_span"))
    # docs = nlp.pipe(citing_spans)
    docs = citing_spans
    normalize = lambda doc: str(doc).strip()
    for i, doc in enumerate(docs):
        # citing_span_sentences = map(normalize, doc.sents)
        citing_span_sentence = [doc]
        # window = non_standard_span_window_sizes.get(i, WINDOW_SIZE)
        start = result[i]["paper"]["in_text"].pop("start")
        end = result[i]["paper"]["in_text"].pop("end")
        window = result[i]["paper"]["in_text"].pop("span_window")
        # print(citing_spans[i])
        citing_sentence_info = get_sentence(citing_spans[i], window, sentence_window=SENTENCE_WINDOW) # , sentences=list(citing_span_sentences))
        result[i]["paper"]["in_text"] |= {
            "start" : window - citing_sentence_info["sentence_start_idx"],
            "end"   : window + (end - start) - citing_sentence_info["sentence_start_idx"],
            "context_start": window - citing_sentence_info["context_start_idx"],
            "context_end": window + (end - start) - citing_sentence_info["context_start_idx"],
        }
        result[i]["sentence"] = citing_sentence_info["sentence"]
        result[i]["context"] = citing_sentence_info["context"]


    return result


def get_spans(matches):
    for match in matches:
        match_group = [i for i, v in enumerate(match.groups()) if v is not None]
        match_group = match_group[0] + 1 if match_group else 0
        yield match.span(match_group)


def get_citing_paper_idxs(sentence):
    """
    Determines where in the sentence the authors refer to their own
    work by looks for common phrases author use to describe their work.

    Only matches one phrase
    """
    sentence = sentence.lower()
    # matches determiners and possessives (to be replaced with [paper]'s)
    pos_patterns = [
        r"\b(our) (?:\w+ )?(?:\w+ )?approach\b",
        r"\b(our) (?:\w+ )?analysis\b",
        r"\b(our) (?:\w+ )?datasets?\b",
        r"\b(our) (?:\w+ )?experiments?\b",
        r"\b(our) (?:\w+ )?implementations?\b",
        r"\b(our) (?:\w+ )?llms?\b",
        r"\b(our) (?:\w+ )?lmms?\b",
        r"\b(our) (?:\w+ )?models?\b",
        r"\b(our) (?:\w+ )?methods?\b",
        r"\b(our) (?:\w+ )?one\b",
        r"\b(our) (?:\w+ )?study\b",
        r"\b(our) (?:\w+ )?studies\b",
        r"\b(our) (?:\w+ )?systems?\b",
        r"\b(our) (?:\w+ )?works?\b",

        r"\b(my) (?:\w+ )?(?:\w+ )?approach\b",
        r"\b(my) (?:\w+ )?analysis\b",
        r"\b(my) (?:\w+ )?datasets?\b",
        r"\b(my) (?:\w+ )?experiments?\b",
        r"\b(my) (?:\w+ )?implementations?\b",
        r"\b(my) (?:\w+ )?llms?\b",
        r"\b(my) (?:\w+ )?lmms?\b",
        r"\b(my) (?:\w+ )?models?\b",
        r"\b(my) (?:\w+ )?methods?\b",
        r"\b(my) (?:\w+ )?one\b",
        r"\b(my) (?:\w+ )?study\b",
        r"\b(my) (?:\w+ )?studies\b",
        r"\b(my) (?:\w+ )?systems?\b",
        r"\b(my) (?:\w+ )?works?\b",

        r"\bour\b",
        r"\bours\b",
        # r"\bmy\b", # might be too many false negatives
        # r"\bmine\b" # overlaps with "mine" as in "text mine"
    ]

    det_patterns = [
        r"\bthis (?:(?!previous )\w+ )?article\b",
        r"\bthis (?:(?!previous )\w+ )?paper\b",
        r"\bthis (?:current )?work\b",
        r"\bthis line of work\b",
        r"\b(this) (?:(?!previous )\w+ )?algorithm\b",
        r"\b(this) (?:(?!previous )\w+ )?approach\b",
        r"\b(this) (?:(?!previous )\w+ )?assumption\b",
        r"\b(this) (?:(?!previous )\w+ )?model\b",
        r"\b(this) (?:(?!previous )\w+ )?method\b",
        r"\b(this) (?:(?!previous )\w+ )?study\b",
        r"\b(this) (?:(?!previous )\w+ )?findings?\b",

        r"\b(these) (?:\w+ )?(?:(?!previous )\w+ )?models\b",
        r"\b(these) (?:(?!previous )\w+ )?extensions\b"
        r"\b(these) (?:(?!previous )\w+ )?algorithms\b",
        r"\b(these) (?:(?!previous )\w+ )?approachs\b",
        r"\b(these) (?:(?!previous )\w+ )?assumptions\b",
        r"\b(these) (?:(?!previous )\w+ )?models\b",
        r"\b(these) (?:(?!previous )\w+ )?methods\b",
        r"\b(these) (?:(?!previous )\w+ )?studies\b",
        r"\b(these) (?:(?!previous )\w+ )?findings\b",

        r"\b(the) (?:proposed |present |current )algorithm\b",
        r"\b(the) (?:proposed |present |current |presented )?(?:(?!previous )\w+ )?approach\b",
        r"\b(the) (?:experimental |present |current )findings\b",
        r"\b(the) (?:above |current |present |proposed )model\b",
        r"\b(the) (?:proposed |present |current )methods?\b",
        r"\b(the) (?:present |current |above |below )system\b",
        r"\b(the) (?:proposed |present |current |above |below )study\b",
        r"\b(the) (?:experimental |present |current )results\b",
        r"\b(the) (?:present |current )work\b",
        r"\b(the) (?:present |current |end-to-end learning )framework\b",

        r"^(this) generalization[s|d]",
        r"^(this) annotation scheme",
    ]

    # matches subjects
    subj_patterns = [
        r"\bwe\b",
        r"(?:^|, |\. |that )i\b(?!\.)",
        r"\bus\b",

        r"^(it) achieve[s|d]\b",
        r"^(it) use[s|d]\b",
        r"^(it) is comparable\b",
        r"^(it) is (?:very )?different\b",
        r"^(it) is (?:very )?similar\b",
        r"^(it) is trained\b",
        r"^(this) generalize[s|d]\b",
        r"^(this) stands\b",
        r"^(this) is just another heuristic\b",
        r"^(this) is (?:very )? similar\b",
        r"^(this) is (?:very )? different\b",
        r"^(this) is better\b",

        # noun phrases would be helpful
        r"^(CULG) adopts\b",
        r"\b(C 2) improves\b",
        r"\b(FVN) extends the Vector-Quantised\b",
        r"\b(GRN) keeps the original graph\b"
        r"\b(HATEMOJICHECK) directly builds\b"
        r"\b(TIGER) does contain\b"
    ]

    loc_patterns = [
        r"\bhere\b",
    ]

    # First check if there are any patterns with "our" or "my" and if there aren't check for
    # patterns like "this work" or "the current work"
    spans = []
    pattern = "|".join(pos_patterns)
    matches = list(re.finditer(pattern, sentence))

    if matches:
        # returns the match of the last group (if there are groups), otherwise the whole string
        spans.extend([(s, "pos") for s in get_spans(matches)])
    else:
        # If there isn't any "our" or "my", then look for "the", "this", and "these"
        pattern = "|".join(det_patterns)
        matches = list(re.finditer(pattern, sentence))
        if matches:
            spans.extend([(s, "det") for s in get_spans(matches)])

    pattern = "|".join(subj_patterns)
    matches = list(re.finditer(pattern, sentence))

    if matches:
        spans.extend([(s, "subj") for s in get_spans(matches)])


    pattern = "|".join(loc_patterns)
    matches = list(re.finditer(pattern, sentence))
    if matches:
        spans.extend([(s, "subj") for s in get_spans(matches)])

    return spans if spans else None

    def test_get_sentence():
    x =  "This is a test citing sentence. It includes a few sentences. On either side of the. Sentence with the citation "\
    "[9]. To test if we are actually. Implementing the get_sentence method correctly. This is a very important test."
    citation_idx = 112

    output_1 = get_sentence(x, 112)
    assert output_1 ==  (
        'Sentence with the citation [9].',
        84,
        ['On either side of the.',
        'Sentence with the citation [9].',
        'To test if we are actually.'],
    )
    output_3 = get_sentence(x, 112, 3)
    assert output_3 == (
        'Sentence with the citation [9].',
        84,
        ['This is a test citing sentence.',
        'It includes a few sentences.',
        'On either side of the.',
        'Sentence with the citation [9].',
        'To test if we are actually.',
        'Implementing the get_sentence method correctly.',
        'This is a very important test.'])