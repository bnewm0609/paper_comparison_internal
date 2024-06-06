import sys
import os
import json
import requests
import time
import logging

from typing import List, Dict

import itertools
from concurrent.futures import ThreadPoolExecutor

import openai
from openai import OpenAI

from value_prompts import *

PAPER_QA_API_URL = "https://s2-labs-paper-qa.apps.allenai.org/api/queryPaperCorpusId/"

S2_API_KEY = os.environ.get('S2_API_KEY')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def openai_call(model: str, prompt: str, response_format: str, temperature=1.0, max_tokens=256) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, 
        {"role": "user", "content": prompt},
    ]
    try:
        completion = ""
        if response_format == "json":
            completion = openai_client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            completion = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    except openai.APIConnectionError as e:
        retry_time = 10
        logging.info("The server could not be reached. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return openai_call(prompt, temperature, max_tokens)
    
    except openai.RateLimitError as e:
        retry_time = 10
        logging.info("A 429 status code was received; we should back off a bit.")
        time.sleep(retry_time)
        return openai_call(prompt, temperature, max_tokens)

    except openai.APIStatusError as e:
        logging.info("Another non-200-range status code was received")
        logging.error(e.status_code)
        logging.error(e.response)
        return "OpenAI failure"

    except Exception as e:
        return "Unknown failure while querying OpenAI"
    
    return completion.choices[0].message.content


def get_value_from_abstract(question: str, corpus_id: str):
    # Retrieve abstract for provided corpus ID
    # headers = {"x-api-key": S2_API_KEY}
    # query = f"https://api.semanticscholar.org/graph/v1/paper/CorpusId:{corpus_id}?fields=title,abstract"
    # response = None
    # retry_num = 0
    # while response is None or response.status_code == '429':
    #     try:
    #         response = requests.get(query, headers=headers)
    #         response_content = json.loads(response.content.decode('utf-8'))
    #     except Exception as e:
    #         print(e)
    #     retry_num += 1
    #     time.sleep(retry_num * 5)
    cur_paper = paper_set[corpus_id]
    title = cur_paper["title"] if "title" in cur_paper else None
    abstract = cur_paper["abstract"] if "abstract" in cur_paper and cur_paper["abstract"] else None
    prompt = VALUE_GENERATION_FROM_ABSTRACT + f"Paper title:{title}\nPaper abstract: {abstract}\nQuestion: {question}\nAnswer:"
    value = openai_call("gpt-3.5-turbo", prompt, "str")
    return value


def run_paper_qa(question: str, corpus_id: str):
    @staticmethod
    def _fp(f, d=3):
        return round(f, d)

    @staticmethod
    def _box2string(box):
        return f"{_fp(box['top'])},{_fp(box['left'])},{_fp(box['height'])},{_fp(box['width'])},{box['page']}"
    
    payload = {
            "prompt": question,
            "corpusId": corpus_id,
            "ui": "Nora",
            "tid": "Nora",
            "userId": "Nora",
    }

    try:
        response = requests.post(
            f"{PAPER_QA_API_URL}/{corpus_id}",
            json=payload,
            timeout=1000,
        ) 
        # If paperQA returns a response, save the answer and excerpts
        if response.status_code == 200:
            response = response.json()
            response_simplified = {
                "question": response.get("question"),
                "answer": response.get("response"),
                "corpusId": response.get("corpusId"),
                "evidenceExcerpts": [
                    {
                        "section_heading": v.get("heading"),
                        "text": v.get("text"),
                        "page": v.get("page"),
                        "boundingBoxes": ";".join(
                            map(_box2string, v.get("bboxs"))
                        ),
                    }
                    for v in response.get("supports", [])
                ],
                "source": "full-text",
            }
        # If paperQA says the question is unanswerable, set the cell value to N/A and excerpts to empty list
        elif response.status_code == 204:
            response_simplified = {
                "question": question,
                "answer": "N/A", 
                "corpusId": corpus_id, 
                "evidenceExcerpts": [],
                "source": "full-text",
            }
        # If paperQA is unable to return any output, we either don't have full-text for the paper or SPP didn't work.
        # In such cases, we try to generate a value from the abstract and record this in the response
        else:
            response = get_value_from_abstract(question=question, corpus_id=corpus_id)
            response_simplified = {
                "question": question,
                "answer": response,
                "corpusId": corpus_id,
                "source": "abstract",
            }
    except Exception as e:
        response_simplified = {"error": f"Exception while querying PaperQA endpoint: {str(e)}"}
    return response_simplified

def generate_value_suggestions(columns_to_populate, corpus_ids) -> Dict:
    cell_values = {}

    # For every column to be populated, run value extraction on all corpusIDs provided.
    for column in columns_to_populate:
        # cid = column["id"]
        # name = column["name"]
        # description = column["description"]

        # Construct a query for paperQA-based value extraction using column description.
        paperqa_query = f"From the provided paper full-text, can you extract {column}?"
        # TODOv2: Add this back if/when we have experiments to run column descriptions
        # if description != '':
        #     paperqa_query += f"{name} can be described as {description}"

        # Run paperQA on full-texts for value extraction first.
        # If full-text is not available, back off to run value extraction from abstracts.
        # TODO: Parallelize paperQA calls
        raw_values = {}
        MAX_THREADS = 5
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            responses = list(executor.map(run_paper_qa, itertools.repeat(paperqa_query), corpus_ids))
            raw_values = {y: x["answer"] if "answer" in x else x["error"] for x,y in zip(responses, corpus_ids)}
        # for corpus_id in corpus_ids:
        #     response = run_paper_qa(paperqa_query, corpus_id)
        #     if "answer" in response:
        #         raw_values[corpus_id] = response["answer"]
        #     else:
        #         # TODO: Does this make sense?
        #         raw_values[corpus_id] = response["error"]

        # TODOv2: Add this back if we have any experiments where columns already have values
        # After extracting all new values for a given column, check if current_table_state
        # contains any existing values in that column. If yes, collect those values.
        # existing_values = []
        # if current_table_state:
        #     for k,v in current_table_state.items():
        #         if k[1] == cid:
        #             existing_values.append(v)

        # Run an additional prompting step to make all newly extracted values look consistent
        prompt_input = "\nRelevant Information: " + json.dumps([raw_values[x] for x in corpus_ids])

        # TODOv2: Add this back if we have any experiments where columns already have values
        # If values already existed in the column, provide them as examples during consistency prompting, otherwise run zero-shot
        # if not existing_values:
        #     prompt_input = VALUE_CONSISTENCY_PROMPT_ZS + prompt_input + "\n\nOutput:"
        # else:
        #     prompt_input = VALUE_CONSISTENCY_PROMPT_FS + "Current Values: " + str(existing_values) + prompt_input + "\n\nOutput:"
        prompt_input = VALUE_CONSISTENCY_PROMPT_ZS + prompt_input + "\n\nOutput:"
        final_values = openai_call("gpt-4-turbo", prompt_input, "json", max_tokens=3000)
        final_values = json.loads(final_values)["values"]

        # TODO: Change reshaping to dump output in our format
        # Reshape final values according to the format expected by NORA handler
        cell_values[column] = {}
        for k,v in zip(corpus_ids, final_values):
            cell_values[column][k] = v

    return cell_values


# Read in papers and tables associated with subsampled metric_validation_0 set on which we will run eval
paper_file = open("../metric_validation_0_papers.jsonl")
table_file = open("../metric_validation_0_tables.jsonl")

paper_set = {}
for line in paper_file:
    data = json.loads(line)
    paper_set[data["corpus_id"]] = data
print(f"Read in metadata for {len(paper_set)} papers...")

tables_to_generate = {}
for line in table_file:
    data = json.loads(line)
    tables_to_generate[data["tabid"]] = [x["corpus_id"] for x in data["row_bib_map"]]
print(f"Running value generation for {len(tables_to_generate)} tables...") 

# Choose a schema generation setting to run value generation for 
schema_folder = sys.argv[1]
model = sys.argv[2]

# For each table to generate, read in column outputs from Yoonjoo's 
# generations for the chosen setting
missing_tables = []
for i, tabid in enumerate(list(tables_to_generate.keys())[:10]):
    print(f"Running value generation for table {i} ({tabid})")
    if not os.path.exists(os.path.join(schema_folder, tabid, model, "ours_outputs", "try_0.json")):
        missing_tables.append(tabid + "_" + model)
        continue
    schema_file = open(os.path.join(schema_folder, tabid, model, "ours_outputs", "try_0.json"))
    full_data = json.loads(schema_file.read())
    schema = full_data[0]["schema"]
    
    corpus_ids = tables_to_generate[tabid]
    final_values = generate_value_suggestions(columns_to_populate=schema, corpus_ids=corpus_ids)
    full_data[0]["table"] = final_values
    out_file = open(os.path.join(schema_folder, tabid, model, "ours_outputs", "try_0_with_values.json"), "w")
    json.dump(full_data, out_file)
    out_file.close()
    

# TODO: Add any error handling/retry policy as needed??