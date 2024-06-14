import sys
import os
import json

import time
import logging
import openai
from openai import OpenAI

# System prompt
SYSTEM_PROMPT = """
You are an intelligent and precise assistant that can understand the contents of research papers. 
You are knowledgable on different fields and domains of science, in particular computer science. 
You are able to interpret research papers, create questions and answers, and compare multiple papers.
"""

# Prompt to generate column description based on column name and caption
CAPTION_PROMPT = """
A user is making a table for a scholarly paper that contains information about multiple papers and compares these papers. 
This table contains a column called [COLUMN]. Please write a  brief definition for this column.

Here is the caption for the table: [CAPTION].

Definition: 
"""

# Prompt to generate column description based on column name, caption name and in-text references
ALL_CONTEXT_PROMPT = """
A user is making a table for a scholarly paper that contains information about multiple papers and compares these papers. 
This table contains a column called [COLUMN]. Please write a  brief definition for this column.

Here is the caption for the table: [CAPTION].

Following is some additional information about this table: [IN_TEXT_REF].

Definition: 
"""

QUESTION_PROMPT = "Rewrite this description as a one-line question."

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def openai_call(model: str, prompt: str, temperature=0, max_tokens=256) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, 
        {"role": "user", "content": prompt},
    ]
    try:
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        column_definition = completion.choices[0].message.content
        messages += [
            {"role": "assistant", "content": column_definition},
            {"role": "user", "content": QUESTION_PROMPT},
        ]
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
        print(e)
        return "Unknown failure while querying OpenAI"
    
    return completion.choices[0].message.content

# Load caption and in-context reference data for entire dataset
context_file = open("../metric_validation_0_dataset_with_ics.jsonl")
context_data = {}
for line in context_file:
    cur_table = json.loads(line)
    context_data[cur_table["_table_hash"]] = cur_table

# Load all table data for entire dataset
table_file = open("../metric_validation_0_tables.jsonl")
tables = {}
for line in table_file:
    cur_table = json.loads(line)
    tables[cur_table["tabid"]] = cur_table

# Get column names, captions and in-text references for each table to be evaluated for value generation
table_data = {}
for id in list(tables.keys()):
    table_data[id] = {}
    table_data[id]["schema"] = list(tables[id]["table"].keys())
    table_data[id]["caption"] = context_data[id]["caption"]
    table_data[id]["in_text_ref"] = "\n".join([x["text"].replace("\n", " ") for x in context_data[id]["in_text_ref"]])

setting = sys.argv[1]
out_file = open(sys.argv[2], "w")
query_data = {}
for i, id in enumerate(list(tables.keys())):
    print(f"Generating queries for table {i}")
    for column in table_data[id]["schema"]:
        query = ""
        if setting == "caption":
            query = CAPTION_PROMPT.replace("[COLUMN]", column).replace("[CAPTION]", table_data[id]["caption"])
        elif setting == "all":
            query = ALL_CONTEXT_PROMPT.replace("[COLUMN]", column).replace("[CAPTION]", table_data[id]["caption"]).replace("[IN_TEXT_REF]", table_data[id]["in_text_ref"])
        final_query = openai_call("gpt-4-turbo", query)
        query_data[id] = final_query
json.dump(query_data, out_file)
out_file.close()