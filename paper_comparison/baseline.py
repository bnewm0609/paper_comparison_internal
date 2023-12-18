import json
import requests
import openai
openai.api_key = 'OPENAI_API_KEY'

# load prompt
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data


# Generation
url = 'https://api.openai.com/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer {}'.format(openai.api_key)
}

def generate(tmp_prompt, generation_type, system_intsruction=None):
    if system_intsruction == None:
        prompt  = [
            {'role': 'user', 'content': tmp_prompt}
        ]
    else:
        prompt  = [
            {'role': 'assistant', 'content': system_intsruction},
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
        max_tokens = 1000
        
    data = {
        'messages': prompt,
        'model': 'gpt-4-1106-preview',
        'max_tokens': max_tokens,
        'temperature': temperature
    }

    response = requests.post(url, headers=headers, json=data)
    output = response.json()
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
    
def zero_shot_paper_to_table(prompt_file_path, paper_list):
    template = load_json_file(prompt_file_path)
    paper_text = ""
    for index, paper in enumerate(paper_list):
        paper_text += f'Paper {index+1} title: {paper["title"]}\n'
        paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
    tmp_prompt = template["zero_shot_paper_to_table"]['prompt_max'].format(paper=paper_text)
    print(tmp_prompt)
    res = generate(tmp_prompt, "generation", system_intsruction=template["zero_shot_paper_to_table"]['system_instruction'])
    return res

def paper_to_cc_to_table(prompt_file_path, paper_list):
    template = load_json_file(prompt_file_path)
    paper_text = ""
    for index, paper in enumerate(paper_list):
        paper_text += f'Paper {index+1} title: {paper["title"]}\n'
        paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
    tmp_prompt = template["zero_shot_paper_to_cc"]['prompt_max'].format(paper=paper_text)
    print(tmp_prompt)
    cc = generate(tmp_prompt, "generation", system_intsruction=template["zero_shot_paper_to_cc"]['system_instruction'])
    paper_cc = ""
    paper_cc += paper_text + "\n" + "Comparison and contrast statements:\n" + cc
    combined_prompt = template["zero_shot_cc_to_table"]['prompt'].format(paper_cc=paper_cc)
    print(combined_prompt)
    table = generate(combined_prompt, "generation", system_intsruction=template["zero_shot_cc_to_table"]['system_instruction'])
    return table

def multiple_papers_to_scheme_to_table(prompt_file_path, paper_list):
    # generate questions that can be answered by the paper
    template = load_json_file(prompt_file_path)
    paper_text = ""
    for index, paper in enumerate(paper_list):
        paper_text += f'Paper {index+1} title: {paper["title"]}\n'
        paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
    tmp_prompt = template["zero_shot_paper_to_scheme"]['prompt'].format(paper=paper_text)
    print(tmp_prompt)
    scheme_qs = generate(tmp_prompt, "generation", system_intsruction=template["zero_shot_paper_to_scheme"]['system_instruction'])
    return scheme_qs

def one_paper_to_scheme_to_table(prompt_file_path, paper_list):
    # generate questions that can be answered by the paper
    template = load_json_file(prompt_file_path)
    scheme_dict = {}
    for index, paper in enumerate(paper_list):
        scheme_dict[paper["title"]] = {}
        paper_text = ""
        paper_text += f'Paper {index+1} title: {paper["title"]}\n'
        paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
        tmp_prompt = template["zero_shot_onepap_to_scheme"]['prompt'].format(paper=paper_text)
        print(tmp_prompt)
        scheme_qs = generate(tmp_prompt, "generation", system_intsruction=template["zero_shot_onepap_to_scheme"]['system_instruction'])
        scheme_dict[paper["title"]]["questions"] = format_to_json(scheme_qs)
    
    return scheme_dict

# paper_list = [{"paperid":"paper1", "title": "Decontextualization: Making Sentences Stand-Alone",
#                         "abstract": "Models for question answering, dialogue agents, and summarization often interpret the meaning of a sentence in a rich context and use that meaning in a new context. Taking excerpts of text can be problematic, as key pieces may not be explicit in a local window. We isolate and define the problem of sentence decontextualization: taking a sentence together with its context and rewriting it to be interpretable out of context, while preserving its meaning. We describe an annotation procedure, collect data on the Wikipedia corpus, and use the data to train models to automatically decontextualize sentences. We present preliminary studies that show the value of sentence decontextualization in a user-facing task, and as preprocessing for systems that perform document understanding. We argue that decontextualization is an important subtask in many downstream applications, and that the definitions and resources provided can benefit tasks that operate on sentences that occur in a richer context."},
#               {"paperid":"paper2", "title": "A Question Answering Framework for Decontextualizing User-facing Snippets from Scientific Documents",
#                          "abstract": "Many real-world applications (e.g., note taking, search) require extracting a sentence or paragraph from a document and showing that snippet to a human outside of the source document. Yet, users may find snippets difficult to understand as they lack context from the original document. In this work, we use language models to rewrite snippets from scientific documents to be read on their own. First, we define the requirements and challenges for this user-facing decontextualization task, such as clarifying where edits occur and handling references to other documents. Second, we propose a framework that decomposes the task into three stages: question generation, question answering, and rewriting. Using this framework, we collect gold decontextualizations from experienced scientific article readers. We then conduct a range of experiments across state-ofthe-art commercial and open-source language models to identify how to best provide missingbut-relevant information to models for our task. Finally, we develop QADECONTEXT, a simple prompting strategy inspired by our framework that improves over end-to-end prompting. We conclude with analysis that finds, while rewriting is easy, question generation and answering remain challenging for today’s models."},
#               {"paperid":"paper3", "title": "Concise Answers to Complex Questions: Summarization of Long-form Answers",
#                          "abstract": "Long-form question answering systems provide rich information by presenting paragraph-level answers, often containing optional background or auxiliary information. While such comprehensive answers are helpful, not all information is required to answer the question (e.g. users with domain knowledge do not need an explanation of background). Can we provide a concise version of the answer by summarizing it, while still addressing the question? We conduct a user study on summarized answers generated from state-of-the-art models and our newly proposed extract-and-decontextualize approach. We find a large proportion of long-form answers (over 90%) in the ELI5 domain can be adequately summarized by at least one system, while complex and implicit answers are challenging to compress. We observe that decontextualization improves the quality of the extractive summary, exemplifying its potential in the summarization task. To promote future work, we provide an extractive summarization dataset covering 1K long-form answers and our user study annotations. Together, we present the first study on summarizing long-form answers, taking a step forward for QA agents that can provide answers at multiple granularities."}]


# def paper_to_scheme_to_table(paper_list):
#     # generate questions that can be answered by the paper
#     paper_text = ""
#     for index, paper in enumerate(paper_list):
#         paper_text += f'Paper {index+1} title: {paper["title"]}\n'
#         paper_text += f'Paper {index+1} abstract: {paper["abstract"]}\n\n'
#     print(template["zero_shot_paper_to_table"]['prompt'])
#     tmp_prompt = template["zero_shot_paper_to_scheme"]['prompt'].format(paper=paper_text)
#     scheme_qs = generate(tmp_prompt, "generation", system_intsruction=template["zero_shot_paper_to_scheme"]['system_instruction'])
    
#     paper_cc = ""
#     paper_cc += paper_text + "\n" + "Questions\n" + scheme_qs
#     combined_prompt = template["zero_shot_scheme_to_table"]['prompt'].format(paper_cc=paper_cc)
#     table = generate(combined_prompt, "generation", system_intsruction=template["zero_shot_cc_to_table"]['system_instruction'])
    
#     # Fill in the values in the table
#     return res

