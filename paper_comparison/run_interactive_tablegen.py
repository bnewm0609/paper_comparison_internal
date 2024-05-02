import os
import json
import random
from typing import Any
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

from paper_comparison.generation import make_paper_list_input, generate, validate_table


EDIT_PROMPT_TEMPLATE = {
    "system_instruction": "You are an intelligent and precise assistant that can understand the contents of research papers. You are knowledgable on different fields and domains of science, in particular computer science. You are able to interpret research papers, create questions and answers, and compare multiple papers.",
    "prompt": "Imagine the following scenario: A user is iteratively making a table for a scholarly paper that contains information about multiple papers and compares these papers. To compare and contrast the papers, the user provides a preliminary comparison table in JSON format as well as the title and abstract of each paper. Your task is the following: Given the current table JSON, paper information and a description of the edits the user wants to make to the table, you should output an updated table in JSON format to reflect these edits. Note that the JSON object should continue to follow the format:\n\n```json\n{{\n  \"<aspect 1>\": [\"<comparable attribute within the aspect 1>\", \"<comparable attribute within the aspect 1>\", ...],\n  ...\n}}\n```\n\n[TABLE]\n\n[PAPER INFO]\n\n[USER INSTRUCTION]\n\nUpdated Table:",
    "parse_str": "```json"
}


def initiate_task(query: str, table: dict):
    """ Start a new task by displaying a target query and pre-generated table to the user.
    
        Args:
            query (str): Query indicating the information need that needs to be addressed via the table
            table (dict): Json containing data for the pre-cached starter table
    """
    table = pd.DataFrame(table)
    print("\nStarting new task!\n")
    print(f"\nUser Query: {query}\n")
    print(table.to_markdown())
    return


def get_user_instruction(instruction: str, table: dict, paper_info: str) -> str:
    """ Generate prompt input to be sent to LLM to execute user-requested table edits.

        Args:
            instruction (str): Brief description of the edits to the table requested by the user
            table (dict): Json containing data for the current table state
            paper_info (str): String containing all aper titles and abstracts
    """
    prompt = EDIT_PROMPT_TEMPLATE["prompt"].replace("[TABLE]", json.dumps(table))
    prompt = prompt.replace("[PAPER INFO]", paper_info)
    prompt = prompt.replace("[USER INSTRUCTION]", instruction)
    return prompt


def generate_updated_table(prompt: str, table: dict, paper_list: list[dict]) -> dict:
    """ Send instruction, current table and paper information to LLM to generate updated table.
        Also validate format of updated table (and retry generation if needed).

        Args:
            instruction (str): Brief description of the edits to the table requested by the user
            table (dict): Json containing data for the current table state
            paper_list (list[dict]): Dictionary containing original paper list data to be used for output validation
    """
    retry_count = 0
    while retry_count < 5:
        # Run generate() function from Yoonjoo's pipeline to leverage API error handling
        updated_table = generate(prompt, "gpt4", "generation", "json", EDIT_PROMPT_TEMPLATE)
        # Validate table output format using Yoonjoo's output parsing logic. Column number set to 0 to bypass that check.
        is_valid, error_type = validate_table(table, paper_list, 0)
        # For invalid table, update number of retries and try again. If table is valid, stop generating
        if not is_valid:
            retry_count += 1
        else:
            return updated_table
    return None


def conduct_interaction_step(instruction: str, table: dict, paper_info: str, paper_list: list[dict]):
    """ Conduct one interaction step in which the given table is edited by an LLM as per user's instructions
        and user feedback about the LLM's instruction following ability is collected. This consists of:
        1. Constructing a prompt to request table edits from an LLM
        2. Calling an LLM and validating the format of the updated table
        3. Displaying the updated table to the user
        4. Collecting user rating and optional text feedback about edit quality
    
        Args:
            instruction (str): Brief description of the edits to the table requested by the user
            table (dict): Json containing data for the current table state
            paper_info (str): String containing all paper titles and abstracts 
            paper_list (list[dict]): Dictionary containing original paper list data to be used for output validation
    """
    # Contains get_updated_table() (include output validation), display_updated_table(), get_user_feedback()
    # Construct a prompt to request table edits from LLM
    prompt = get_user_instruction(instruction, table, paper_info)

    # Run table editing until a valid output table is produced or maximum number of retries is hit
    updated_table = generate_updated_table(prompt, table, paper_list)
    
    # Display updated table
    print(f"Updated Table:\n{pd.DataFrame(updated_table).to_markdown()}\n")

    # Collect user feedback for the updated table
    user_rating = input("Does the updated table reflect your requested edits? (y/n/p)\n")
    user_feedback = input("Would you like to provide additional feedback about this turn? (If not, press enter)\n")

    return updated_table, user_rating, user_feedback


def run_table_construction_task(queries: dict, tables_dir: str, gold_tables_dir: str) -> str:
    """ Main entry point for interactive table generation.
        Randomly selects a user query and pre-generated table to display to the user.
        Allows the user to request edits to the table which are fulfilled by an LLM.
        Records feedback about LLM's ability to perform edits correctly.
        When user is satisfied with final table, stores the complete interaction log and return task ID.

        Args:
            queries (dict): Dictionary containing query IDs and queries for tasks not completed yet
            tables_dir (str): Path to directory containing pre-cached starting tables for all tasks
            gold_tables_dir (str): Path to directory containing gold data (paper titles+abstracts) for all tasks
    """
    query_ids = list(queries.keys())
    # Randomly choose a task to display to the user.
    task_id = random.sample(query_ids, 1)[0]

    # Load corresponding query, pre-cached table and paper data to display
    cur_query = queries[task_id]
    cur_table = json.loads(open(os.path.join(tables_dir, f"{task_id}.json")).read())["table"]
    cur_gold_data = json.loads(open(os.path.join(gold_tables_dir, f"{task_id}.json")).read())

    # Concatenate all paper data (titles and abstracts) to provide as additional context
    # whenever the user asks for any edits to be made to the table.
    # Uses prompt formatting strategy from Yoonjoo's pipeline code. 
    paper_text = ""
    for p_idx, paper in enumerate(cur_gold_data["row_bib_map"]):
        paper_text = make_paper_list_input(paper_text, p_idx, paper, "abs", "multiple")
    paper_text = f"[Start of the Paper Content]\n{paper_text}[End of the Paper Content]"

    # Start the task by displaying a query and starter table
    initiate_task(cur_query, cur_table)

    # Start a logging object to store interaction data
    interaction_log = {}

    user_instruction = ''
    step = 0
    # Continue the interaction while the user still wants edits to be made to the table
    # TODO: Do we want to set a hard limit on #edits after which interaction always ends?
    while user_instruction.lower() != 'done':
        if user_instruction != '':
            updated_table, user_rating, user_feedback = conduct_interaction_step(user_instruction, cur_table, paper_text, cur_gold_data["row_bib_map"])
            # Log output of interaction step
            interaction_log[step] = {
                "input_table": cur_table,
                "output_table": updated_table,
                "user_instruction": user_instruction,
                "user_rating": user_rating,
                "user_feedback": user_feedback
            }
            # Increment step counter and replace table state
            step += 1
            cur_table = updated_table
        user_instruction = input("If you would like to make further edits to this table, briefly describe the edit in natural language (e.g., add a column about X). If no more edits are needed, please type 'done'.\nInstruction: ")
        
    # After interaction is complete, store the complete interaction log
    json.dump(interaction_log, open(os.path.join(interaction_dir, f'{task_id}_interaction.json'), 'w'))
    return task_id

if __name__ == "__main__":

    # TODO: Add path to correct stored user queries, gold tables directory (for paper titles+abstracts) and tables.
    tables_dir = "../sample_tables/"
    gold_tables_dir = "../sample_tables/"
    query_file = "../sample_queries.json"
    query_set = json.loads(open(query_file).read())

    # Path to directory to store interaction logs.
    # If this directory does not exist, create it.
    interaction_dir = "../interaction_logs/"
    if not os.path.exists(interaction_dir):
        os.makedirs(interaction_dir)

    task_flag = "y"
    # Continue running the script till more tasks are available
    # and the user is willing to continue doing tasks.
    while task_flag == "y" and list(query_set.keys()):
        # Ask the user if they want to start a new task. If not, end execution.
        task_flag = input("Would you like to start a new task? (y/n): ")
        if task_flag == "n":
            break
        # If yes, call the function that runs a new task.
        completed_task_id = run_table_construction_task(query_set, tables_dir, gold_tables_dir)
        # Delete ID of completed task to avoid repeating it again.
        # TODO: Record this so that tasks are also not repeated in the next session.
        del query_set[completed_task_id]