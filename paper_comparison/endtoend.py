import json
from typing import Any,Optional, Sequence
import openai
import requests
import itertools
import random

from omegaconf import DictConfig

from paper_comparison.types.table import Table
openai.api_key = 'OPENAI_API_KEY'

class BaseEndToEnd:
    def __init__(self, args: DictConfig):        
        pass
    
    def __call__(self, args, data) -> list[Table]:
        return []

class BaselineEndToEnd:
    def __init__(self, args: DictConfig):
        self.template = self.load_json_file(args.endtoend.prompt_path)
        
    def __call__(self, args, data) -> list[Table]:
        baseline_tables = []
        for index, sample in enumerate(data):
            if "pap_to_tab" in args.endtoend.baseline_type:
                tab = self.format_to_json(self.zero_shot_paper_to_table(sample["x"], intro=False))
                baseline_tables.append(Table(tabid=str(index), schema=set(tab.keys()),values=tab,type="pap_to_tab"))
            if "cc_to_tab" in args.endtoend.baseline_type:
                tab = self.format_to_json(self.paper_to_cc_to_table(sample["x"], intro=False))
                baseline_tables.append(Table(tabid=str(index), schema=set(tab.keys()),values=tab,type="cc_to_tab"))
            if "multi_scheme" in args.endtoend.baseline_type:
                tab = self.one_paper_to_scheme_to_table(sample["x"], answer=True, intro=False)
                baseline_tables.append(Table(tabid=str(index), schema=set(tab.keys()),values=tab,type="multi_scheme"))
            
        return baseline_tables

    def load_json_file(self, file_path: str) -> Any:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return json_data
    
    def generate(self, tmp_prompt:str, generation_type:str, system_instruction=Optional[str])->str:
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer {}'.format(openai.api_key)
        }
        if system_instruction == None:
            prompt  = [
                {'role': 'user', 'content': tmp_prompt}
            ]
        else:
            prompt  = [
                {'role': 'assistant', 'content': system_instruction},
                {'role': 'user', 'content': tmp_prompt}
            ]
        if generation_type == "verification":
            temperature = 0.3
            max_tokens = 30
        elif generation_type == "specificity":
            temperature = 0.3
            max_tokens = 2000
        else:
            temperature = 1
            max_tokens = 4000
            
        data = {
            'messages': prompt,
            'model': 'gpt-4-1106-preview',
            'max_tokens': max_tokens,
            'temperature': temperature
        }
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            try:
                output = response.json()
                # Process your output here
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
        else:
            print("Error:", response.status_code, response.text)
        
        if 'choices' in output:
            for choice in output['choices']:
                message = choice['message']
                if message['role'] == 'assistant':
                    explanation = message['content']
                else:                
                    print("This was the first problem:", explanation)
        else:
            print("This was the second roblem:", explanation)
        return explanation

    def format_to_json(self, text: str) -> Any:
        if text.find('[System]') != -1:
            text = text[text.find('[System]') + 8:]
        
        if text.find('```json') != -1:
            start = text.find('```json') + 7
            end = text.rfind('```') - 1
        elif text.find('[') != -1 and (text.find('{') == -1 or text.find('{') > text.find('[')):
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
            # print("ERROR:", e)
            # print("TEXT:", text)
            # print("JSON TEXT:", json_text)
            raise e
        
    def zero_shot_paper_to_table(self, paper_list: Sequence[Any], intro: bool=False)->str:
        paper_text = ""
        if intro:
            for index, paper in enumerate(paper_list):
                paper_text += f'Paper {index+1} title: {paper["title"]}\n'
                if paper["introduction"] == "None":
                    paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
                else:
                    paper_text += f'Paper {index+1} introduction: {paper["introduction"]}\n\n'
        else:
            for index, paper in enumerate(paper_list):
                paper_text += f'Paper {index+1} title: {paper["title"]}\n'
                paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
        tmp_prompt = self.template["zero_shot_paper_to_table"]['prompt_max'].format(paper=paper_text)
        
        res = self.generate(tmp_prompt, "generation", system_instruction=self.template["zero_shot_paper_to_table"]['system_instruction'])
        return res

    def paper_to_cc_to_table(self, paper_list: Sequence[Any], intro: bool=False)->str:
        paper_text = ""
        if intro:
            for index, paper in enumerate(paper_list):
                paper_text += f'Paper {index+1} title: {paper["title"]}\n'
                if paper["introduction"] == "None":
                    paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
                else:
                    paper_text += f'Paper {index+1} introduction: {paper["introduction"]}\n\n'
        else:
            for index, paper in enumerate(paper_list):
                paper_text += f'Paper {index+1} title: {paper["title"]}\n'
                paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
        tmp_prompt = self.template["zero_shot_paper_to_cc"]['prompt_max'].format(paper=paper_text)
        cc = self.generate(tmp_prompt, "generation", system_instruction=self.template["zero_shot_paper_to_cc"]['system_instruction'])
        paper_cc = ""
        paper_cc += paper_text + "\n" + "Comparison and contrast statements:\n" + cc
        combined_prompt = self.template["zero_shot_cc_to_table"]['prompt'].format(paper_cc=paper_cc)
        
        table = self.generate(combined_prompt, "generation", system_instruction=self.template["zero_shot_cc_to_table"]['system_instruction'])
        return table

    def multiple_papers_to_scheme_to_table(self, paper_list: Sequence[Any], answer: bool=False)->str:
        # generate questions that can be answered by the paper
        paper_text = ""
        for index, paper in enumerate(paper_list):
            paper_text += f'Paper {index+1} title: {paper["title"]}\n'
            paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
        tmp_prompt = self.template["zero_shot_paper_to_scheme"]['prompt'].format(paper=paper_text)
        scheme_qs = self.generate(tmp_prompt, "generation", system_instruction=self.template["zero_shot_paper_to_scheme"]['system_instruction'])
        if answer:
            concatenated_schemes = ""
            paper_scheme = ""
            for qs in scheme_qs.values():
                concatenated_schemes += '\n'.join(qs["questions"])
            paper_scheme += "Papers\n" + paper_text + "Questions\n" + concatenated_schemes
            combined_prompt = self.template["zero_shot_value_generation"]['prompt'].format(schemes=paper_scheme)
            table = self.generate(combined_prompt, "generation", system_instruction=self.template["zero_shot_value_generation"]['system_instruction'])
            return table
        else:
            return scheme_qs

    def one_paper_to_scheme_to_table(self, paper_list: Sequence[Any], intro: bool=False, answer: bool=False)->str:
        # generate questions that can be answered by the paper
        scheme_dict = {}
        for index, paper in enumerate(paper_list):
            scheme_dict[paper["title"]] = {}
            paper_text = ""
            paper_text += f'Paper {index+1} title: {paper["title"]}\n'
            paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
            tmp_prompt = self.template["zero_shot_onepap_to_scheme"]['prompt'].format(paper=paper_text)
            scheme_qs = self.generate(tmp_prompt, "generation", system_instruction=self.template["zero_shot_onepap_to_scheme"]['system_instruction'])
            scheme_dict[paper["title"]]["questions"] = self.format_to_json(scheme_qs)
            
        concatenated = list(itertools.chain(*(v["questions"] for v in scheme_dict.values())))
        # concatenated = list(itertools.chain(scheme_dict.values()))
        sampled_qs = random.sample(concatenated, 20)

        if answer:
            concatenated_schemes = ""
            paper_scheme = ""
            paper_text = ""

            tmp_concatenated_schemes = '\n'.join(sampled_qs)
                # tmp_concatenated_schemes += '\n'.join(qs["questions"])
            for index, q in enumerate(sampled_qs):
                concatenated_schemes += f'{index}. {q}\n'
                
            # make a dictionary that has a question as a key and a list of answers as a value
            final_dict = {}
            for qs in tmp_concatenated_schemes.split("\n"):
                final_dict[qs] = {}
            # make an answer for all the question from each paper
            for index, paper in enumerate(paper_list):
                if intro:
                    variable = f'Paper Introduction\n{paper["introduction"]}' if paper["introduction"] != "None" else f'Paper Abstract\n{paper["abstract"]}'           
                    paper_scheme += f'Paper Title\n{paper["title"]}\n\n {variable}\n\n'
                    paper_scheme += f'Question\n{concatenated_schemes}\n'
                    tmp_prompt = self.template["zero_shot_value_generation"]['prompt'].format(schemes=paper_scheme)
                    tmp_answer = self.generate(tmp_prompt, "generation", system_instruction=self.template["zero_shot_value_generation"]['system_instruction'])
                    answer = self.format_to_json(tmp_answer)
                    #  fill in the value of the question
                    for q_id, qs in enumerate(tmp_concatenated_schemes.split("\n")):
                        final_dict[qs][f'paper_{index}'] = list(answer.values())[q_id]  
                else:
                    paper_scheme += f'Paper Title\n{paper["title"]}\n\n Paper Abstract\n{paper["abstract"]}\n\n'
                    paper_scheme += f'Question\n{concatenated_schemes}\n'
                    tmp_prompt = self.template["zero_shot_value_generation"]['prompt'].format(schemes=paper_scheme)
                    tmp_answer = self.generate(tmp_prompt, "generation", system_instruction=self.template["zero_shot_value_generation"]['system_instruction'])
                    answer = self.format_to_json(tmp_answer)
                    #  fill in the value of the question
                    for q_id, qs in enumerate(tmp_concatenated_schemes.split("\n")):
                        final_dict[qs][f'paper_{index}'] = list(answer.values())[q_id]  
            return final_dict
        else:
            return scheme_dict

class ComputedOutputsEndToEnd(BaseEndToEnd):
    def __call__(self, args, data) -> list[Table]:
        
        table_values = {
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
        return [
            Table(
                tabid="0",
                schema=set(table_values.keys()),
                values=table_values,
            )
        ]
    
        
class DebugAbstractsEndToEnd(BaseEndToEnd):
    def __call__(self, args, data) -> list[Table]:
        table_values = {
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
        return [
            Table(
                tabid="0",
                schema=set(table_values.keys()),
                values=table_values,
            )
        ]
    

class PrecomputedOutputsEndToEnd(BaseEndToEnd):
    def __call__(self, args, data) -> list[Table]:
        with open(args.endtoend.path) as f:
            table_values = json.load(f)

        # the baselines (baseline_paper_to_table_max.json, baseline_paper_to_cc_tab_max.json) contain only the tables
        # so, we don't need to extract them, but for Our algorithm (ours_output_decontext.json) the file contains other
        # info
        if "final_table" in table_values:
            table_values = table_values["final_table"]
            for attribute in table_values:
                del table_values[attribute]["type"]
                del table_values[attribute]["presup"]

        return [
            Table(
                tabid="0",
                schema=table_values.keys(),
                values=table_values,
            )
        ]


class OracleEndToEnd(BaseEndToEnd):
    """Returns the gold tables"""

    def __call__(self, args, data) -> list[Table]:
        return [sample["y"] for sample in data]


def load_endtoend(args: DictConfig):
    # breakpoint()
    if args.endtoend.name == "debug_abstracts":
        return DebugAbstractsEndToEnd(args)
    elif args.endtoend.name == "precomp_outputs":
        return PrecomputedOutputsEndToEnd(args)
    elif args.endtoend.name == "oracle":
        return OracleEndToEnd(args)
    elif args.endtoend.name == "comp_outputs":
        return ComputedOutputsEndToEnd(args)
    elif args.endtoend.name == "baseline_outputs":
        return BaselineEndToEnd(args)
    return BaseEndToEnd(args)
