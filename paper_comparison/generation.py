import json
import requests
import openai
from typing import Any, Optional, Sequence, Dict, List
import os

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def format_table(table_question):
    table = {}
    for key, value in table_question.items():
        table[value["question"]] = {}
        for k, v in value.items():
            if k != "question":
                table[value["question"]][k] = [v]
    return table

def str_to_json(text, parse_str):
    try:
        json_str = text.split(parse_str)[1].strip()
        return json.loads(json_str)
    except:
        # potential parse_str: ["[\JSON]"]
        potential_parse_str = ["[/JSON]", "{/JSON}"]
        for p_str in potential_parse_str:
            try:
                json_str = text.split(parse_str)[1].split(p_str)[0].strip()
                # json_str = text.split(p_str)[1].strip()
                return json.loads(json_str)
            except:
                try:
                    json_str = fix_json_str(json_str)
                    return json.loads(json_str)
                except:
                    continue
        print('Failed to parse with the given parse_str')
        return text

def fix_json_str(json_str):
    template = load_json_file("./data/prompt_ver2.json")["json_formatting"]
    tmp_prompt = template['prompt'].format(input_info=json_str)
    api_key = os.environ['OPENAI_KEY']
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer {}'.format(api_key)
    }
    try:
        data = {
            'messages': [
                {'role': 'assistant', 'content': template["system_instruction"]},
                {'role': 'user', 'content': tmp_prompt}
            ],
            'model': 'gpt-4-1106-preview',
            'max_tokens': 4000,
            'temperature': 0.3
        }
        response = requests.post(url, headers=headers, json=data)
        # print(response)
        response.raise_for_status()  # Raises a HTTPError if the response contains an HTTP error status code
        output = response.json()
        if 'choices' in output:
            for choice in output['choices']:
                message = choice['message']
                if message['role'] == 'assistant':
                    explanation = message['content']
        return explanation.split(template["parse_str"])[1].strip()
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")        

def format_to_json(text):
    if text.find('[') != -1 and (text.find('{') == -1 or text.find('{') > text.find('[')):
        # list
        start = text.find('[')
        end = text.rfind(']')
    else:
        # dict
        start = text.find('{')
        end = text.rfind('}')
    json_text = text[start:end+1]
    try:
        return json.loads(json_text)
    except Exception as e:
        print("ERROR:", e)
        print("TEXT:", text)
        print("JSON TEXT:", json_text)
        raise e

def make_paper_list_input(paper_text:str, index:int, paper:Dict, source:str, paper_loop:str) -> str:
    if paper_loop == "single":
        paper_text += f'Paper title: {paper["title"]}\n'
        if source == "intro":
            if paper["introduction"] == "None":
                paper_text += f'Paper abstract: {paper["abstract"]}\n\n'
            else:
                paper_text += f'Paper introduction: {paper["introduction"]}\n\n'
        elif source == "full":
            paper_text += f'Paper abstract: {paper["abstract"]}\n\n'
        else:
            paper_text += f'Paper abstract: {paper["abstract"]}\n\n'

    else:
        if source == "intro":
            paper_text += f'Paper {index+1} title: {paper["title"]}\n'
            if paper["introduction"] == "None":
                paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
            else:
                paper_text += f'Paper {index+1} introduction: {paper["introduction"]}\n\n'
        elif source == "full":
            paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
        else:
            paper_text += f'Paper {index+1} title: {paper["title"]}\naper {index+1} abstract: {paper["abstract"]}\n\n'
    return paper_text

def merge_tables(tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    # {"id": int(index), "tabid": str(tab_id), "text": table, "error_type": error_type, "error_num": error_num}
    
    merged_table = {"id": tables[0]["id"], "tabid": tables[0]["tabid"], "caption": tables[0]["caption"], 
                    "schema": [], "table": {}, "gold_col":0, "predicted_col_num":0, "error_num": 0, }
    for table in tables:
        if "text" in list(table.keys()):
            merged_table = {"id": table["id"], "tabid": table["tabid"], "text": table["text"], 
                            "error_type": table["error_type"], "error_num": merged_table["error_num"] + table["error_num"]} 
        else:
            merged_table["schema"].extend(table["schema"])
            merged_table["table"].update(table["table"])
            merged_table["gold_col"] += table["gold_col"]
            merged_table["predicted_col_num"] += table["predicted_col_num"]
            merged_table["error_num"] += table["error_num"]
            merged_table["type"] = table["type"]
    return merged_table
            
def generate(tmp_prompt, model_type, generation_type, data_type, template=None):
    explanation = ""
    if model_type == "gpt4":
        api_key = os.environ['OPENAI_KEY']
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(api_key)
        }
        model = 'gpt-4-1106-preview'    
    elif model_type == "mixtral":
        api_key = os.environ['TOGETHER_API_KEY']
        url = 'https://api.together.xyz'
        model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(api_key)
        }
    elif model_type == "llama":
        api_key = os.environ['TOGETHER_API_KEY']
        url = 'https://api.together.xyz/v1/chat/completions'
        model = "meta-llama/Llama-2-70b-chat-hf"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(api_key)
        }
    
    try:
        if template["system_instruction"] == None:
            prompt  = [
                {'role': 'user', 'content': tmp_prompt}
            ]
        else:
            prompt  = [
                {'role': 'assistant', 'content': template["system_instruction"]},
                {'role': 'user', 'content': tmp_prompt}
            ]
        if generation_type == "verification":
            temperature = 0.3
            max_tokens = 30
        elif generation_type == "specificity":
            temperature = 0.3
            max_tokens = 1000
        else:
            temperature = 1
            max_tokens = 3000
        
        if model_type == "gpt4":   
            data = {
                'messages': prompt,
                'model': model,
                'max_tokens': max_tokens,
                'temperature': temperature
            }
            response = requests.post(url, headers=headers, json=data)
            # print(response)
            response.raise_for_status()  # Raises a HTTPError if the response contains an HTTP error status code
            output = response.json()
            if 'choices' in output:
                for choice in output['choices']:
                    message = choice['message']
                    if message['role'] == 'assistant':
                        explanation = message['content']
        elif model_type == "mixtral":
            client = openai.OpenAI(api_key=api_key, base_url=url)
            chat_completion = client.chat.completions.create(
                model=model,
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            explanation = chat_completion.choices[0].message.content
        elif model_type == "llama":
            data = {
                'prompt': f"[INST] {prompt} [/INST]",
                'model': model,
                'max_tokens': 650,
                'temperature': temperature,
                'stop': ["[/INST]", "</s>"]
            }
            response = requests.post(url, headers=headers, json=data)
            print(response)
            response.raise_for_status()
            output = response.json()
            if 'choices' in output:
                for choice in output['choices']:
                    message = choice['message']
                    if message['role'] == 'assistant':
                        explanation = message['content'] 
            print("generation is completed")  
        if data_type == "list":
            return explanation.strip()
        else:
            return str_to_json(explanation, template["parse_str"])  
            
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")  