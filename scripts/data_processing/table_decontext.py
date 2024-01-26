### For an example of the output of this script, see
# `data/arxiv_tables_2308_high_quality/dataset_autocontext.jsonl`
from argparse import ArgumentParser
import json
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


ROOT_DIR = Path(__file__).parent.parent.parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer():
    mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
    mistral_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        load_in_8bit=True,
        # torch_dtype=torch.float16,
        # device_map=DEVICE,
    )
    mistral_model.config.pad_token_id = mistral_model.config.eos_token_id
    return mistral_model, mistral_tokenizer


def is_numeric(value):
    """
    Filters out values that contain some common numeric values that don't need context.
    They include units like 'K' for thousand, 'M' for million and others.
    These are *not* comprehensive, and are curated only on the tables evaluated
    so far from the high quality 2308 papers.
    """
    value = value.replace("below", "<")
    value = re.sub(r"[<$∼~∼\s]", "", value.lower().strip())
    units_regex = r"[<-]?\d+[km]"
    time_regex = r"(\d+(?:hrs?|hours?))?(\d+(?:ms?|mins?|minutes?))?(\d+(?:s|sec|seconds?))?"
    freq_regex = r"\d+[gm]hz"
    if (
        re.match(units_regex, value) is not None
        or any(re.match(time_regex, value).groups())
        or re.match(freq_regex, value) is not None
    ):
        return True
    # this is dumb, but easy and fast
    try:
        float(value.replace(",", "").replace("-", ""))
        return True
    except ValueError:
        return False


def is_binary(value):
    value = value.strip()
    return value.lower() in {"yes", "no"} or value in {"✘", "✗", "×", "✔", "✓"}


def is_na(value):
    value = re.sub("[∼~\s]", "", value.lower())
    return value in {"-", "–", "-"}


def create_glossary_instructions(instructions_data):
    instruction = f"""\
Based on the following text from a scientific paper, answer the question about the table that follows:

Title: {instructions_data['title']}

Abstract: {instructions_data['abstract']}

Section: {instructions_data['section_name']}
{instructions_data['table_section']}

---
Based on the above text, in the context of the following table, what does {instructions_data['glossary_query']} refer to? Answer in a single sentence. If the answer is not clear just write 'unanswerable'.
Table: {instructions_data['table_hash']}
{instructions_data['table']}\
"""
    return instruction


def query_mistral(model, tokenizer, prompt):
    generated_ids = []
    inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(DEVICE)
    try:
        generated_ids = model.generate(inputs, max_new_tokens=1000, do_sample=True, num_return_sequences=1)
        response = tokenizer.batch_decode(generated_ids[:, inputs.shape[1] :], skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError:
        # for debugging
        print("oom:", inputs.shape, flush=True)
        raise torch.cuda.OutOfMemoryError
    finally:
        # to avoid taking up gpu memory
        del inputs
        del generated_ids
        torch.cuda.empty_cache()

    return response


def listify(list_paragraph):
    """Formats a list paragraph (like sometimes seen at the end of an intro section) as a list for the prompt."""
    return " * " + "\n * ".join(list_paragraph.strip().split("\n\n"))


def get_section_with_table(dataset_table, full_text):
    """
    Gets the first section from the full text that mentions the table.
    E.g. "As shown in Tab. 1, ..."

    This method relies on the tralics/unarxiv processing functions which
    are supposed to pull out the table references from the . This doesn't alwayws
    work though---not all tables are linked and sometimes mentions to other figures
    are linked. Not totally sure why. Investigations into this have to happen on the
    data curation side.
    """

    section_names = []
    for paragraph in full_text["body_text"]:
        for ref_span in paragraph["ref_spans"]:
            if ref_span["ref_id"] == dataset_table["_full_text_table_hash"]:
                section_names.append(paragraph["section"])

    relevant_sections = {sn: [] for sn in section_names}
    for paragraph in full_text["body_text"]:
        if paragraph["section"] in relevant_sections:
            if paragraph["content_type"] == "paragraph":
                relevant_sections[paragraph["section"]].append(paragraph["text"])
            elif paragraph["content_type"] == "list":
                relevant_sections[paragraph["section"]].append(listify(paragraph["text"]))
            # else:
            #     print(paragraph['content_type'])

    # Just return the first section for now. Eventually, if the table is referenced
    # in more than one section, we might want to return all of the sections
    table_section = "\n".join(list(relevant_sections.values())[0])
    return list(relevant_sections.keys())[0], table_section


def create_glossary(dataset_table, glossary_keys, prompt_data, model, tokenizer):
    """
    Creates the glossary for the given table by generating context for the given glossary_keys.

    dataset_table [dict]: contains the table along with table metadata (eg what paper the table came from)
    glossary_keys [list[Tuple]]: each element is a tuple of (column header, table cell value) that should be
        decontextualized.
    prompt_data [dict]: used for the prompt to the model. Includes
    model, tokenizer: model and tokenizer to prompt
    """
    # Convert the table to a pandas dataframe for easier masking and the str() method
    table_df = pd.DataFrame(dataset_table["table_json"]["table_dict"])
    glossary = {}
    for scheme, glossary_key in glossary_keys:
        # Don't repeat glossary terms
        if (scheme, glossary_key) in glossary:
            continue
        print((scheme, glossary_key), flush=True)

        # Format the key to the glossary for the model prompt
        if scheme == glossary_key:
            glossary_query = f"'{glossary_key}'"
        else:
            glossary_query = f"'{glossary_key}' in the column '{scheme}'"

        # mask out the columns that aren't being asked about and add caption
        table = str(table_df[["References", scheme]]) + "\n"
        table += "Caption: " + dataset_table["table_unfiltered"]["caption"]

        prompt_data = prompt_data | {
            "glossary_query": glossary_query,
            "table_hash": dataset_table["_full_text_table_hash"],
            "table": table,
        }

        instruction = create_glossary_instructions(prompt_data)

        # If you get an OOM error, remove a paragraph off the end of the
        # table section and try again until you don't get an OOM error
        while True:
            prompt = [{"role": "user", "content": instruction}]
            try:
                if model is None:
                    response = ["test"]
                else:
                    response = query_mistral(model, tokenizer, prompt)
                break
            except torch.cuda.OutOfMemoryError:
                print("OOM num chars:", len(instruction))
                prompt_data["table_section"] = "\n".join(prompt_data["table_section"].split("\n")[:-1])
                instruction = create_glossary_instructions(prompt_data)
        glossary[(scheme, glossary_key)] = response[0]
    return instruction, glossary


def main():
    argp = ArgumentParser()
    argp.add_argument("--debug", action="store_true")
    argp.add_argument(
        "--out_path", default=ROOT_DIR / "data/arxiv_tables_2308_high_quality/dataset_autocontext_DEBUG.jsonl"
    )
    args = argp.parse_args()
    # load in the data:
    # Has the gold tables that we're going to decontextualize
    with open(ROOT_DIR / "data/arxiv_tables_2308_high_quality/dataset.jsonl") as f:
        dataset_2308_high_quality_sample = [json.loads(line.strip()) for line in f]

    # Has the full texts associated with those tables along with the tables themselves
    with open(ROOT_DIR / "data/arxiv_tables_2308_high_quality/full_texts_with_tables.jsonl") as f:
        full_texts_2308_high_quality = [json.loads(line) for line in f]
        full_texts_2308_high_quality_map = {ft["paper_id"]: ft for ft in full_texts_2308_high_quality}

    # next, load the model:
    if args.debug:
        model, tokenizer = None, None
    else:
        model, tokenizer = load_model_and_tokenizer()

    # Now, create the glosssary
    start_i = 0
    for dataset_table_i, dataset_table in enumerate(dataset_2308_high_quality_sample[start_i:]):
        dataset_table_i += start_i
        print("\n----------\n")
        print(dataset_table_i, dataset_table["paper_id"], dataset_table["_full_text_table_hash"])
        full_text = full_texts_2308_high_quality_map[dataset_table["paper_id"]]

        # first, create the glossary keys. Assume that every header (aka scheme) and value
        # in the table needs to be decontextualized. Then, filter out the ones that don't
        # In particular, we don't want to add context for table values that are
        # numbers, binary (checks and x's), or missing because the context for the *header* of
        # those columns will provide the needed information. Also don't try to add context for
        # items in the "References" columns because they are just a bunch of hashes or corpus_ids
        # which don't actually appear in the paper.
        glossary_keys = [
            (scheme.strip(), str(entry).strip())
            for scheme, column in dataset_table["table_json"]["table_dict"].items()
            for entry in [scheme] + column
            if scheme != "References"
            and not is_numeric(str(entry))
            and not is_na(str(entry))
            and not is_binary(str(entry))
        ]

        # next, get the section containing the table, the title and the abstract
        section_name, table_section = get_section_with_table(dataset_table, full_text)
        input_data = {
            "title": full_text["title"],
            "abstract": full_text["abstract"]["text"],
            "section_name": section_name,
            "table_section": table_section,
        }
        sample_instruction, glossary = create_glossary(dataset_table, glossary_keys, input_data, model, tokenizer)

        # the glossary is saved with the "<~>" between the column and the term
        # because there's no native tuple -> json serializer and I didn't write one
        dataset_table["context_autogenerated"] = {
            "glossary": {"<~>".join(k): v for k, v in glossary.items()},
            "inputs": {
                "section_name": section_name,
                "instruction_sample": sample_instruction,  # include this for debugging purposes
            },
        }

    # Save the generations in a separate place for now for debugging purposes.
    # When we're confident with the decontextualizations and format and everything,
    # we can overwrite the dataset.jsonl file
    with open(args.out_path, "w") as f:
        for line in dataset_2308_high_quality_sample:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
