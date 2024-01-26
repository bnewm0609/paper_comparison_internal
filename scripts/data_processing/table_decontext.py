import json
import re
import requests

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEVICE = "cuda"
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", cache_dir="/gscratch/xlab/blnewman/models/transformers/")
mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
mistral_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    cache_dir="/gscratch/xlab/blnewman/models/transformers/",
    load_in_8bit=True,
    # torch_dtype=torch.float16,
    # device_map=DEVICE,
)
mistral_model.config.pad_token_id = mistral_model.config.eos_token_id

with open("../data/arxiv_tables_2308_high_quality/dataset.jsonl") as f:
    dataset_2308_high_quality_sample = [json.loads(line.strip()) for line in f]

with open("../data/arxiv_tables_2308_high_quality/full_texts_with_tables_v2.jsonl") as f:
    full_texts_2308_high_quality = [json.loads(line) for line in f]
    full_texts_2308_high_quality_map = {ft["paper_id"]: ft for ft in full_texts_2308_high_quality}

with open("../data/arxiv_tables_2308_high_quality/papers_expanded.jsonl") as f:
    papers_2308_high_quality = [json.loads(line) for line in f]

metadata_map = {paper['arxiv_id']: paper for paper in papers_2308_high_quality if 'arxiv_id' in paper}


def is_numeric(value):
    # value = value.lower()
    value = value.replace("below", "<")
    value = re.sub(r"[<$∼~∼\s]", "", value.lower().strip())
    units_regex = r"[<-]?\d+[km]"
    time_regex = r"(\d+(?:hrs?|hours?))?(\d+(?:ms?|mins?|minutes?))?(\d+(?:s|sec|seconds?))?"
    freq_regex = r"\d+[gm]hz"
    if re.match(units_regex, value) is not None or any(re.match(time_regex, value).groups()) or re.match(freq_regex, value) is not None:
        return True
    # this is dumb, but easy and fast
    try:
        float(value.replace(",", "").replace("-", ""))
        return True
    except ValueError:
        return False

def is_binary(value):
    value = value.strip()
    return value.lower() in {'yes', 'no'} or value in {'✘', '✗', '×', '✔', '✓'}

def is_na(value):
    value = re.sub("[∼~\s]", "", value.lower())
    return value in {"-", '–', '-'}

def create_glossary(dataset_table, glossary_keys, title, abstract, section_name, table_section):
    table_df = pd.DataFrame(dataset_table['table_json']['table_dict'])
    # glossary = {("Dataset", "Dataset"): "In the context of the table in the text, 'Dataset' refers to a specific collection of video data along with associated annotations, used for the tasks of video quality assessment (VQA) or video classification."}
    glossary = {}
    for scheme, glossary_key in glossary_keys:
        # Avoid repeating glossary terms
        if (scheme, glossary_key) in glossary:
            continue
        print((scheme, glossary_key))
        if scheme == glossary_key:
            glossary_query = f"'{glossary_key}'"
        else:
            glossary_query = f"'{glossary_key}' in the column '{scheme}'"
    
        # mask out the columns that aren't being asked about:
        table = str(table_df[["References", scheme]]) + "\n"
        table += "Caption: " + dataset_table['table_unfiltered']['caption']
    
        instruction = f"""\
Based on the following text from a scientific paper, answer the question about the table that follows:

Title: {title}

Abstract: {abstract}

Section: {section_name}
{table_section}

---
Based on the above text, in the context of the following table, what does {glossary_query} refer to? Answer in a single sentence. If the answer is not clear just write 'unanswerable'.
Table: {dataset_table['_full_text_table_hash']}
{table}\
"""
        
        prompt = [{"role": "user", "content": instruction}]
        inputs = mistral_tokenizer.apply_chat_template(prompt, return_tensors="pt").to(DEVICE)
        generated_ids = mistral_model.generate(inputs, max_new_tokens=1000, do_sample=True, num_return_sequences=1)
        response = mistral_tokenizer.batch_decode(generated_ids[:, inputs.shape[1]:], skip_special_tokens=True)
        # response = ["test"]
        glossary[(scheme, glossary_key)] = response[0]
        del inputs
        del generated_ids
        torch.cuda.empty_cache()
    return instruction, glossary


def listify(list_paragraph):
        return " * " + "\n * ".join(list_paragraph.strip().split("\n\n"))

def get_section_with_table(dataset_table, full_text):
    section_names = []
    for paragraph in full_text['body_text']:
        for ref_span in paragraph['ref_spans']:
            # if "DNN-based VQA method" in paragraph["body_text"]:
            #     print(paragraph)
            if ref_span["ref_id"] == dataset_table['_full_text_table_hash']:
                # print(paragraph['content_type'])
                # print(paragraph.keys())
                section_names.append(paragraph['section'])
    
    relevant_sections = {sn: [] for sn in section_names}
    for paragraph in full_text['body_text']:
        if paragraph['section'] in relevant_sections:
            if paragraph['content_type'] == "paragraph":
                relevant_sections[paragraph['section']].append(paragraph['text'])
            elif paragraph['content_type'] == "list":
                relevant_sections[paragraph['section']].append(listify(paragraph['text']))
            # else:
            #     print(paragraph['content_type'])

    # just return the first section for now
    table_section = "\n".join(list(relevant_sections.values())[0])
    return list(relevant_sections.keys())[0], table_section



# contextualization_inputs = {}
start_i = 0
for dataset_table_i, dataset_table in enumerate(dataset_2308_high_quality_sample[start_i:]):
    dataset_table_i += start_i
    print("\n----------\n")
    print(dataset_table_i, dataset_table["paper_id"], dataset_table["_full_text_table_hash"])
    full_text = full_texts_2308_high_quality_map[dataset_table['paper_id']]
    # first, create the unique glossary keys:
    glossary_keys = [
        (scheme.strip(), str(entry).strip())
        for scheme, column in
        dataset_table['table_json']['table_dict'].items()
        for entry in [scheme] + column
        if scheme != "References" and not is_numeric(str(entry)) and not is_na(str(entry)) and not is_binary(str(entry))
    ]
    # glossary_keys_unique = []
    # for key in glossary_keys:
    #     if key not in glossary_keys_unique:
    #         glossary_keys_unique.append(key)

    # next, get the section, title and abstract
    section_name, table_section = get_section_with_table(dataset_table, full_text)
    arxiv_id = re.sub("v\d+", "", dataset_table["paper_id"])
    title = metadata_map[arxiv_id]['title']
    abstract = metadata_map[arxiv_id]['abstract']
    print(section_name)
    print(table_section)
    instruction_sample, glossary = create_glossary(dataset_table, glossary_keys, title, abstract, section_name, table_section)

    dataset_table["context_autogenerated"] = {
        "glossary": {"<~>".join(k): v for k, v in glossary.items()},
        "inputs": {
            "section_name": section_name,
            "instruction_sample": instruction_sample,
        }
    }

    dataset_table["title"] = title
    dataset_table["abstract"] = abstract
    # "title": title,
    # "abstract": abstract,

    # contextualization_inputs[dataset_table["_table_hash"]] = {
    #     "glossary": {"<~>".join(k): v for k, v in glossary.items()},
    #     "inputs": {
    #         "title": title,
    #         "abstract": abstract,
    #         "section_name": section_name,
    #         "table_section": table_section,
    #         "instruction_sample": instruction_sample,
    #     }
    # }