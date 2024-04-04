import json
from typing import Any, Optional, Sequence, Dict, List
import openai
import requests
import itertools
import random

from omegaconf import DictConfig

from paper_comparison.types.table import Table
from paper_comparison.generation import load_json_file, generate, divide_column_num, make_paper_list_input, format_to_json, merge_tables, validate_table, baseline_create_json_format_template, mark_length_error, ours_create_json_format_template

class BaseEndToEnd:
    def __init__(self, args: DictConfig):        
        pass
    
    def __call__(self, args, data) -> List[Table]:
        return []

class BaselineEndToEnd:
    def __init__(self, args: DictConfig):
        self.template = load_json_file(args.endtoend.prompt_path)
        
    def __call__(self, args, sample, tab_id, index, column_num, gold_caption) -> Table:
        table_list = []
        for i in range(args.endtoend.num_commonality):
            if "single_call" in args.endtoend.baseline_type:
                print(column_num * len(sample["x"]))
                if column_num * len(sample["x"]) < args.endtoend.max_length:    # when the number of columns and papers is enough to be generated in one table
                    baseline_table = self.baseline_tab_gen(sample["x"], args.endtoend.model_type, index, tab_id, column_num, gold_caption, source="abstract")
                    baseline_table["error_counts"] = mark_length_error(baseline_table["error_counts"])
                else:
                    print("The number of columns and papers is large")
                    column_list = divide_column_num(column_num, len(sample["x"]), args.endtoend.max_length)
                    baseline_table_set = []
                    for partial_column_num in column_list:
                        partial_table = self.baseline_tab_gen(sample["x"], args.endtoend.model_type, index, tab_id, partial_column_num, gold_caption, source="abstract")
                        print("\npartial_table", partial_table)
                        baseline_table_set.append(partial_table)
                    baseline_table = merge_tables(baseline_table_set)
                    baseline_table["error_counts"] = mark_length_error(baseline_table["error_counts"])
                # table_list.append(baseline_table)
                
            if "pap_to_tab" in args.endtoend.baseline_type:
                tab = self.format_to_json(self.zero_shot_paper_to_table(sample["x"], intro=False))
                baseline_table = Table(tabid=str(index), schema=set(tab.keys()), values=tab, type="pap_to_tab")
            if "cc_to_tab" in args.endtoend.baseline_type:
                tab = self.format_to_json(self.paper_to_cc_to_table(sample["x"], intro=False))
                baseline_table = Table(tabid=str(index), schema=set(tab.keys()), values=tab, type="cc_to_tab")
            if "multi_scheme" in args.endtoend.baseline_type:
                tab = self.one_paper_to_scheme_to_table(sample["x"], answer=True, intro=False)
                baseline_table = Table(tabid=str(index), schema=set(tab.keys()), values=tab, type="multi_scheme")
            table_list.append(baseline_table)
            
        return table_list
    
    def baseline_tab_gen(
        self, paper_list: List[Dict[str, Any]], model: str, 
        index: int, tab_id: str, column_num: int, 
        gold_caption: Optional[str] = None, source: str = "abs"
    ) -> Dict[str, Any]:
        # when column_num * paper_num is less than 120, generate one table
        # when column_num * paper_num is more than 120, give half column number and generate two tables
        paper_text = ""
        
        # set up error counts
        error_counts = {
            "length_error": 0,
            "json_error": 0,
            "paper_num_error": 0,
            "column_num_error": 0
        }
        
        # make a paper text that will be inputted to the prompt
        for p_idx, paper in enumerate(paper_list):
            paper_text = make_paper_list_input(paper_text, p_idx, paper, source, "multiple")
        paper_text = f"[Start of the Paper Content]\n{paper_text}[End of the Paper Content]"

        # Create JSON foramt template
        if type(gold_caption) == str:
            tmp_prompt = baseline_create_json_format_template(self.template["baseline_paper_to_table"]["prompt_medium"],
                                                 column_num, paper_list, paper_text, gold_caption)
        else:
            tmp_prompt = baseline_create_json_format_template(self.template["baseline_paper_to_table"]["prompt_simple"],
                                                 column_num, paper_list, paper_text)
        merged_table_list = []
        while (error_counts["length_error"] < 5):
            tmp_prompt = baseline_create_json_format_template(self.template["baseline_paper_to_table"]["prompt_simple"],
                                                 column_num, paper_list, paper_text)
            table = generate(tmp_prompt, model, "generation", "json", self.template["baseline_paper_to_table"])  
               
            is_valid, error_type = validate_table(table, paper_list, column_num)
            if is_valid:
                merged_table_list.append(table)
                if len(merged_table_list) > 1:
                    baseline_table = merge_tables(merged_table_list)    
                else: 
                    baseline_table = {
                        "id": int(index), 
                        "tabid": str(tab_id), 
                        "schema": list(table.keys()), 
                        "table": table,
                        "gold_col": column_num, 
                        "predicted_col_num":len(list(table.keys())), 
                        "type": "single_call", 
                        "caption": "N/A", 
                        "error_counts": error_counts
                    }
                break
            elif error_type == "column_num_error":  # generate another table with the remaining number of columns (new_column_num = column_num - len(table.keys()))
                merged_table_list.append(table)
                column_num = column_num - len(table.keys())
                error_counts[error_type] += 1
            else:
                error_counts[error_type] += 1
                print(f"\t\t{tab_id} - Table Error: {error_type}. Total error count: {sum(list(error_counts.values()))}")
                baseline_table = {
                    "id": int(index), 
                    "tabid": str(tab_id), 
                    "text": table, 
                    "error_counts": error_counts
                }
        return baseline_table
        
    def zero_shot_paper_to_table(self, paper_list: Sequence[Any], intro: bool=False) -> str:
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
        sampled_qs = random.sample(concatenated, 20)

        if answer:
            concatenated_schemes = ""
            paper_scheme = ""
            paper_text = ""

            tmp_concatenated_schemes = '\n'.join(sampled_qs)
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
    def __init__(self, args: DictConfig):
        self.template = load_json_file(args.endtoend.prompt_path)

    def __call__(self, args, sample, tab_id, index, column_num, gold_caption = None) -> List[Table]:
        if column_num * len(sample["x"]) < args.endtoend.max_length:
            tables = self.table_prediction(args, tab_id, index, sample["x"], column_num, gold_caption)
        else:
            tables = self.table_prediction(args, tab_id, index, sample["x"], column_num, args.endtoend.max_length, gold_caption) 
        for table in tables:
            table["error_counts"] = mark_length_error(table["error_counts"]) 
        return tables
    
    def find_similarity(self, paper_list: List[Dict[str, Any]], model:str, source: str = "abstract", max: bool = True) -> str:
        paper_text = ""
        for index, paper in enumerate(paper_list):
            paper_text = make_paper_list_input(paper_text, index, paper, source, "multiple")
        if max:
            tmp_prompt = self.template["find_similarity"][f'prompt_abstract'].format(paper=paper_text)
            res = generate(tmp_prompt, model, "generation", "list", system_intsruction=self.template["find_similarity"])
        else:
            tmp_prompt = self.template["find_similarity"][f'prompt_withoutmax'].format(paper=paper_text)
            res = generate(tmp_prompt, model, "generation", "list", self.template["find_similarity"])
        return res
    
    def validate_scheme(self, scheme):
        if isinstance(scheme, dict):
            return True, ""
        elif isinstance(scheme, str): 
            # if there is [JSON] once in the table
            if scheme.strip()[-1] != "}": # length issue - generate once more
                return False, "scheme_length_error"
            else:
                print(scheme)
                return False, "scheme_json_error"
        else:
            print(scheme)
            return False, "scheme_unknown_error"
        
    def generate_commonality_attribute(self, paper_list: List[Dict[str, Any]], model:str, num_column: int, source: str = "abstract") -> Dict[str, Any]:
        error_counts = {
            "scheme_length_error": 0,
            "scheme_json_error": 0,
            "scheme_unknown_error": 0,
        }
        paper_text = ""
        for p_idx, paper in enumerate(paper_list):
            paper_text = make_paper_list_input(paper_text, p_idx, paper, source, "multiple")
        paper_text = f"[Start of the Paper Content]\n{paper_text}[End of the Paper Content]"

        tmp_prompt = self.template["scheme_attribute_generation"][f'prompt_abstract'].format(num_attributes=num_column, input_info=paper_text)
        max_try = 5
        for i in range(max_try):
            res = generate(tmp_prompt, model, "generation", "json", self.template["scheme_attribute_generation"])
            is_valid, error_type = self.validate_scheme(res)
            if is_valid:
                scheme = res
                break
            else:
                error_counts[error_type] += 1
                print(f"\t\tScheme Error: {error_type}. Total error count: {sum(list(error_counts.values()))}")
                scheme = {"text": res, "error_type": error_type, "error_counts": error_counts}
        return scheme, error_counts

    def find_attributes(self, paper_list: List[Dict[str, Any]], model:str, research_topic: str) -> str:
        paper_text = ""
        for index, paper in enumerate(paper_list):
            paper_text += f'Paper {index+1} title: {paper["title"]}\n'
            
        tmp_prompt = self.template["find_attributes"]['prompt2'].format(paper=paper_text, research_topic=research_topic)
        res = generate(tmp_prompt, model, "generation", "list", self.template["find_attributes"])
        return res
        
    def value_generation(
        self, paper_list: List[Dict[str, Any]], model: str, 
        similarity: str, attributes: List[str], attribute_num: int, 
        id: int, tab_id: str, save_type: str, 
        scheme_error_counts: Dict[str, Any]={}, source: str = "abstract", paper_loop: str = "multiple"
    ) -> Dict[str, Any]:
        prompt_type = "value_generation_qa"    # value_generation_qa or value_generation
        temp_template = "[Start of the Paper Content]\n{paper}[End of the Paper Content]\n\n[Start of Table Caption]\n{similarity}\n[End of Table Caption]\n\n[Start of Table Columns]\n{columns}\n[End of Table Columns]\n\n"
        col_names = '\n'.join([f'Column name {index+1}: {att}' for index, att in enumerate(attributes)])
        paper_text = ""
        error_counts = {
            "length_error": 0,
            "json_error": 0,
            "paper_num_error": 0,
            "column_num_error": 0
        }
        for key, value in scheme_error_counts.items():
            error_counts[key] = value
        if paper_loop == "single":
            table = {att: {} for att in attributes[:attribute_num]}
            for index, paper in enumerate(paper_list):
                paper_text = make_paper_list_input(paper_text, index, paper, source, paper_loop)
                input_info = temp_template.format(paper=paper_text, similarity=similarity, columns=col_names)
                combined_prompt = self.template[prompt_type][f'prompt_{paper_loop}_{source}'].format(input_info=input_info)
                row = generate(combined_prompt, model, "generation", "json", self.template[prompt_type])
                # row = str_to_json(output, self.template[prompt_type]["parse_str"])
                for att_index, att in enumerate(table.keys()):
                    table[att][f'paper_{index+1}'] = row[f'column {att_index+1}']
        else:
            for index, paper in enumerate(paper_list):
                paper_text = make_paper_list_input(paper_text, index, paper, source, paper_loop)
            # input_info = temp_template.format(paper=paper_text, similarity=similarity, columns=col_names)
            
            # # Create JSON format template to add to the prompt
            # json_format = {}
            # for att in attributes:
            #     json_format[att] = {}
            #     for i in range(len(paper_list)):
            #         json_format[att][f'paper_{i+1}'] = [f"<value for this column grounded on Paper {i+1}>"]
            # json_format = json.dumps(json_format, indent=2)

            # combined_prompt = self.template[prompt_type][f'prompt_{paper_loop}_{source}'].format(input_info=input_info, json_format=json_format)
            # max_try = 5
            # for i in range(max_try):
        merged_table_list = []
        while(error_counts["length_error"] < 5):
            # combined_prompt = self.template[prompt_type][f'prompt_{paper_loop}_{source}'].format(input_info=input_info, json_format=json_format)
            combined_prompt = ours_create_json_format_template(temp_template, self.template[prompt_type][f'prompt_{paper_loop}_{source}'], 
                                                                paper_text, len(paper_list), similarity, attributes)
            table = generate(combined_prompt, model, "generation", "json", self.template[prompt_type])
            is_valid, error_type = validate_table(table, paper_list, attribute_num)
            if is_valid:
                merged_table_list.append(table)
                if len(merged_table_list) > 1:
                    our_table = merge_tables(merged_table_list)
                else:
                    our_table = {
                        "id": int(id), 
                        "tabid": str(tab_id), 
                        "schema": list(table.keys()), 
                        "table": table, 
                        "caption": similarity, 
                        "gold_col": attribute_num,
                        "predicted_col_num": len(list(table.keys())),
                        "type": save_type, 
                        "error_counts": error_counts
                    }
                    break
            elif error_type == "column_num_error":  # generate another table with the remaining number of columns (new_column_num = column_num - len(table.keys()))
                merged_table_list.append(table)
                attributes = [item for item in attributes if item not in list(table.keys())]
                attribute_num = len(attributes)
                error_counts[error_type] += 1
            else:
                
                error_counts[error_type] += 1
                print(f"\t\t{tab_id} - Table Error: {error_type}. Total error count: {sum(list(error_counts.values()))}")
                print(table)
                our_table = {
                    "id": int(id), 
                    "tabid": str(tab_id),
                    "text": table, 
                    "error_counts": error_counts
                }
        return our_table
        
    def filter_out(self, commonality_attribute_sets: Dict, method: str, num_commonalities: Optional[int] = 1) -> List[Any]:
        commonalities = list(commonality_attribute_sets.keys())
        if len(commonalities) < num_commonalities:
            return commonalities
        else:
            if method == "random":
                # random sampling one of the commonalities
                return random.sample(commonalities, num_commonalities)
                
            elif method == "best":   # need revision: identify elements that are different from each other
                # based on the information of the papers, commonalities, and attributes, return the best commonality index
                # idx = generate_best_commonality(commonality_attribute_sets)
                # commonalities = commonalities[idx]
                return []
        
    def table_prediction(
        self, args, tab_id, 
        id, paper_data: List[Dict[str, Any]], col_num: int, 
        max_length: int = None, gold_caption: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        
        table_set = []
        scheme_set = {} 
        save_type = "{}_{}_{}_{}".format(args.endtoend.model_type, args.endtoend.num_commonality, args.endtoend.attribute_gen_method, args.endtoend.paper_loop)
        if type(gold_caption) != str:   # difficulty: hard
            if args.endtoend.attribute_gen_method == "single_call":  # our current method
                commonality_attribute_sets, scheme_error_counts = self.generate_commonality_attribute(paper_data, args.endtoend.model_type, col_num, source=args.endtoend.scheme_source)
                scheme_valid = False if "text" in list(commonality_attribute_sets.keys()) else True
                commonalities = list(commonality_attribute_sets.keys())
                commonalities = self.filter_out(commonality_attribute_sets, "random", num_commonalities=args.endtoend.num_commonality)
            else:
                commonalities = format_to_json(self.find_similarity(paper_data, args.endtoend.model_type, source=args.endtoend.scheme_source, max=True))
        else:
            commonalities = [gold_caption]

        if scheme_valid:
            # make a table for each commonality
            scheme_set = {} 
            for commonality in commonalities:
                if args.endtoend.attribute_gen_method == "single_call":
                    attributes = commonality_attribute_sets[commonality]
                else:
                    attributes = format_to_json(self.find_attributes(paper_data, args.endtoend.model_type, commonality))
                scheme_set[commonality] = attributes
                if max_length != None:     # when the value generation needs to be divided into multiple calls
                    start = 0
                    merged_table_set = []
                    column_list = divide_column_num(len(attributes), len(paper_data), max_length)
                    for partial_column_num in column_list:
                        partial_attributes = attributes[start:start+partial_column_num]
                        start += partial_column_num
                        tab = self.value_generation(
                            paper_data, args.endtoend.model_type, 
                            commonality, partial_attributes, partial_column_num, 
                            id, tab_id, save_type, scheme_error_counts,
                            args.endtoend.value_source, args.endtoend.paper_loop
                        )
                        if "text" in tab:
                            break
                        merged_table_set.append(tab)
                        
                    table = merge_tables(merged_table_set)
                else:
                    table = self.value_generation(
                        paper_data, args.endtoend.model_type, 
                        commonality, attributes, len(attributes), 
                        id, tab_id, save_type, scheme_error_counts,
                        args.endtoend.value_source, args.endtoend.paper_loop
                    )
                table_set.append(table)
        else:
            print("Scheme generation failed")
            commonality_attribute_sets["id"] = int(id)
            table_set.append(commonality_attribute_sets)
        return table_set
        
class DebugAbstractsEndToEnd(BaseEndToEnd):
    def __call__(self, args, data) -> List[Table]:
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
    elif args.endtoend.name == "ours_outputs":
        return ComputedOutputsEndToEnd(args)
    elif args.endtoend.name == "baseline_outputs":
        return BaselineEndToEnd(args)
    return BaseEndToEnd(args)
