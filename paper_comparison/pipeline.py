import json
import requests
import openai
import ast
import os
openai.api_key = 'OPENAI_API_KEY'

# load prompt
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data

def generate(tmp_prompt, generation_type, system_intsruction=None):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer {}'.format(openai.api_key)
    }
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
        max_tokens = 250
        
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

def extract_presuppositions(question):
    tmp_prompt = template["extract_presupposition"]['prompt']
    tmp_prompt += f'Question: {question}\n'
    tmp_prompt += 'Presupposition:'
    # print(tmp_prompt)
    res = generate(tmp_prompt, "generation", system_intsruction=template["extract_presupposition"]['system_instruction'])
    return res.strip()

def evaluate_presupposition(paper, presupposition):
    res = False
    tmp_prompt = template["evaluate_presupposition"]['prompt']
    tmp_prompt += f'Presupposition: {presupposition}\n'
    tmp_prompt += f'Paper title: {paper["title"]}\n'
    tmp_prompt += f'Paper abstract: {paper["abstract"]}\n'
    tmp_prompt += 'Satisfied:'
    # print(tmp_prompt)
    res = generate(tmp_prompt, "generation", system_intsruction=template["evaluate_presupposition"]['system_instruction'])
    return res.strip()

def generate_answer(paper, question):
    tmp_prompt = template["generate_answer"]['prompt']
    tmp_prompt += f'Question: {question}\n'
    tmp_prompt += f'Paper title: {paper["title"]}\n'
    tmp_prompt += f'Paper abstract: {paper["abstract"]}\n'
    tmp_prompt += 'Answer:'
    # print(tmp_prompt)
    res = generate(tmp_prompt, "generation", system_intsruction=template["generate_answer"]['system_instruction'])
    return res.strip()

def is_repetitive(question, question_list):
    repetitive = False
    tmp_prompt = template["is_repetitive"]['prompt']
    tmp_prompt += f'Question: {question}\n'
    tmp_prompt += f'Question list: {question_list}\n'
    tmp_prompt += 'Repetitive:'
    # print(tmp_prompt)
    res = generate(tmp_prompt, "generation", system_intsruction=template["is_repetitive"]['system_instruction'])
    if res.strip().lower() == "true":
        repetitive = True
    else:
        repetitive = False
    return repetitive

def generate_followup(question, answer):
    tmp_prompt = template["generate_followup"]['prompt']
    tmp_prompt += f'Question: {question}\n'
    tmp_prompt += f'Answer: {answer}\n'
    tmp_prompt += 'Follow-up question:'
    # print(tmp_prompt)
    res = generate(tmp_prompt, "generation", system_intsruction=template["generate_followup"]['system_instruction'])
    return ast.literal_eval(res.strip())

def generate_lowlevel(answer, question):
    tmp_prompt = template["generate_lowlevel"]['prompt']
    tmp_prompt += f'Main question: {question}\n'
    tmp_prompt += f'Answer: {answer}\n'
    tmp_prompt += 'Low-level question:'
    # print(tmp_prompt)
    res = generate(tmp_prompt, "generation", system_intsruction=template["generate_lowlevel"]['system_instruction'])
    return ast.literal_eval(res.strip())

def generate_generic(question):
    tmp_prompt = template["generate_generic"]['prompt']
    tmp_prompt += f'Original Question: {question}\n'
    tmp_prompt += 'Generic questions:'
    print(tmp_prompt)
    res = generate(tmp_prompt, "generation", system_intsruction=template["generate_generic"]['system_instruction'])
    return ast.literal_eval(res.strip())

def single_loop(table_question, table_presupposition, question, paper, q_id, paper_id, round, question_list):
    presupposition = extract_presuppositions(question)
    satisfied = evaluate_presupposition(paper, presupposition)
    table_presupposition[f"presup_{q_id}"]["question"] = question
    table_presupposition[f"presup_{q_id}"]["presup"] = presupposition
    table_question[f"question_{q_id}"]["presup"] = presupposition
    if satisfied.lower() == "true":
        table_presupposition[f"presup_{q_id}"][f"paper_{paper_id}"] =  True
        answer = generate_answer(paper, question)
        table_question[f"question_{q_id}"][f"paper_{paper_id}"] = answer
        
        if len(question_list) > 30:
            print("Too many questions")
        else:
            followup = generate_followup(question, answer)
            lowlevel = generate_lowlevel(answer, question)
            
            for q in followup:
                if (is_repetitive(q, question_list) == False) and (round < 2):
                    question_list.append({"round": round+1, "question": q, "type": "followup"})
                
            for q in lowlevel:
                if (is_repetitive(q, question_list) == False) and (round < 2):
                    question_list.append({"round": round+1, "question": q, "type": "lowlevel"})
    else:
        table_presupposition[f"presup_{q_id}"][f"paper_{paper_id}"] =  False
        if len(question_list) > 30:
                print("Too many questions")
        else:
            generic = generate_generic(question)
            for q in generic:
                if (is_repetitive(q, question_list) == False) and (round < 2):
                    question_list.append({"round": round+1, "question": q, "type": "generic"})

    return table_question, table_presupposition, question_list

def format_table(table_question):
    table = {}
    for key, value in table_question.items():
        table[value["question"]] = {}
        for k, v in value.items():
            if k != "question":
                table[value["question"]][k] = [v]
    return table

def iteration(template, json_file_path, paper_list):
    # Check if JSON file exists
    if os.path.exists(json_file_path):
        # Load data from JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            table_question = data.get('table_question', {})
            table_presupposition = data.get('table_presupposition', {})
            question_list = data.get('question_list', [])
            final_table = data.get('final_table', {})
            q_id = len(final_table)
    else:
        # Initialize data
        table_question = {}
        table_presupposition = {}
        question_list = [{"round": 1, "question": "what does this paper study?", "type": "initial"}, {"round":1, "question":"what is the method this paper proposed?", "type": "initial"}]
        q_id = 0
    
    while len(question_list) > 0:
        if q_id >= 25:
            break
        single_question_set = question_list.pop(0)
        question = single_question_set["question"]
        round = single_question_set["round"]
        table_question[f"question_{q_id}"] = {f"paper_{i}": "" for i in range(len(paper_list))}
        table_question[f"question_{q_id}"]["question"] = question
        table_question[f"question_{q_id}"]["type"] = single_question_set["type"]
        table_presupposition[f"presup_{q_id}"] = {}
        
        for paper_id, paper in enumerate(paper_list):
            print("processing:", q_id, paper_id, question)
            print(len(question_list))
            table_question, table_presupposition, question_list = single_loop(table_question, table_presupposition, question, paper, q_id, paper_id, round, question_list)
        
        final_table = format_table(table_question)
        # Write to JSON file
        with open(f'./data/pilot_1128/ours_output_{q_id}.json', 'w') as f:
            json.dump({"final_table": final_table, "table_question": table_question, "table_presupposition": table_presupposition, "question_list": question_list}, f)
            print(f"Finish writing {q_id}th turn")

        q_id += 1
        
    return table_question, table_presupposition, question_list, final_table

paper_list = [{"paperid":"paper1", "title": "Decontextualization: Making Sentences Stand-Alone",
                        "abstract": "Models for question answering, dialogue agents, and summarization often interpret the meaning of a sentence in a rich context and use that meaning in a new context. Taking excerpts of text can be problematic, as key pieces may not be explicit in a local window. We isolate and define the problem of sentence decontextualization: taking a sentence together with its context and rewriting it to be interpretable out of context, while preserving its meaning. We describe an annotation procedure, collect data on the Wikipedia corpus, and use the data to train models to automatically decontextualize sentences. We present preliminary studies that show the value of sentence decontextualization in a user-facing task, and as preprocessing for systems that perform document understanding. We argue that decontextualization is an important subtask in many downstream applications, and that the definitions and resources provided can benefit tasks that operate on sentences that occur in a richer context."}, 
              {"paperid":"paper2", "title": "A Question Answering Framework for Decontextualizing User-facing Snippets from Scientific Documents", 
                         "abstract": "Many real-world applications (e.g., note taking, search) require extracting a sentence or paragraph from a document and showing that snippet to a human outside of the source document. Yet, users may find snippets difficult to understand as they lack context from the original document. In this work, we use language models to rewrite snippets from scientific documents to be read on their own. First, we define the requirements and challenges for this user-facing decontextualization task, such as clarifying where edits occur and handling references to other documents. Second, we propose a framework that decomposes the task into three stages: question generation, question answering, and rewriting. Using this framework, we collect gold decontextualizations from experienced scientific article readers. We then conduct a range of experiments across state-ofthe-art commercial and open-source language models to identify how to best provide missingbut-relevant information to models for our task. Finally, we develop QADECONTEXT, a simple prompting strategy inspired by our framework that improves over end-to-end prompting. We conclude with analysis that finds, while rewriting is easy, question generation and answering remain challenging for today’s models."}, 
              {"paperid":"paper3", "title": "Concise Answers to Complex Questions: Summarization of Long-form Answers", 
                         "abstract": "Long-form question answering systems provide rich information by presenting paragraph-level answers, often containing optional background or auxiliary information. While such comprehensive answers are helpful, not all information is required to answer the question (e.g. users with domain knowledge do not need an explanation of background). Can we provide a concise version of the answer by summarizing it, while still addressing the question? We conduct a user study on summarized answers generated from state-of-the-art models and our newly proposed extract-and-decontextualize approach. We find a large proportion of long-form answers (over 90%) in the ELI5 domain can be adequately summarized by at least one system, while complex and implicit answers are challenging to compress. We observe that decontextualization improves the quality of the extractive summary, exemplifying its potential in the summarization task. To promote future work, we provide an extractive summarization dataset covering 1K long-form answers and our user study annotations. Together, we present the first study on summarizing long-form answers, taking a step forward for QA agents that can provide answers at multiple granularities."}]

template = load_json_file('./prompt.json')
def main(json_file_path, paper_list):   
    table_question, table_presupposition, question_list, final_table=iteration(table_file_path, paper_list)
    return table_question, table_presupposition, question_list, final_table

# hci_paper_list = [{"paperid":"paper1", "title": "Papeos: Augmenting Research Papers with Talk Videos",
#                         "abstract": "Research consumption has been traditionally limited to the reading of academic papers—a static, dense, and formally written format. Alternatively, pre-recorded conference presentation videos, which are more dynamic, concise, and colloquial, have recently become more widely available but potentially under-utilized. In this work, we explore the design space and benefits for combining academic papers and talk videos to leverage their complementary nature to provide a rich and fluid research consumption experience. Based on formative and co-design studies, we present Papeos, a novel reading and authoring interface that allow authors to augment their papers by segmenting and localizing talk videos alongside relevant paper passages with automatically generated suggestions. With Papeos, readers can visually skim a paper through clip thumbnails, and fluidly switch between consuming dense text in the paper or visual summaries in the video. In a comparative lab study (n=16), Papeos reduced mental load, scaffolded navigation, and facilitated more comprehensive reading of papers."},
#               {"paperid":"paper2", "title": "CiteSee: Augmenting Citations in Scientific Papers with Persistent and Personalized Historical Context",
#                          "abstract": "When reading a scholarly article, inline citations help researchers contextualize the current article and discover relevant prior work. However, it can be challenging to prioritize and make sense of the hundreds of citations encountered during literature reviews. This paper introduces CiteSee, a paper reading tool that leverages a user’s publishing, reading, and saving activities to provide personalized visual augmentations and context around citations. First, CiteSee connects the current paper to familiar contexts by surfacing known citations a user had cited or opened. Second, CiteSee helps users prioritize their exploration by highlighting relevant but unknown citations based on saving and reading history. We conducted a lab study that suggests CiteSee is significantly more effective for paper discovery than three baselines. A field deployment study shows CiteSee helps participants keep track of their explorations and leads to better situational awareness and increased paper discovery via inline citation when conducting real-world literature reviews."},
#               {"paperid":"paper3", "title": "CiteRead: Integrating Localized Citation Contexts into Scientific Paper Reading",
#                          "abstract": "When reading a scholarly paper, scientists oftentimes wish to understand how follow-on work has built on or engages with what they are reading. While a paper itself can only discuss prior work, some scientific search engines can provide a list of all subsequent citing papers; unfortunately, they are undifferentiated and disconnected from the contents of the original reference paper. In this work, we introduce a novel paper reading experience that integrates relevant information about follow-on work directly into a paper, allowing readers to learn about newer papers and see how a paper is discussed by its citing papers in the context of the reference paper. We built a tool, called CiteRead, that implements the following three contributions: 1) automated techniques for selecting important citing papers, building on results from a formative study we conducted, 2) an automated process for localizing commentary provided by citing papers to a place in the reference paper, and 3) an interactive experience that allows readers to seamlessly alternate between the reference paper and information from citing papers (e.g., citation sentences), placed in the margins. Based on a user study with 12 scientists, we found that in comparison to having just a list of citing papers and their citation sentences, the use of CiteRead while reading allows for better comprehension and retention of information about follow-on work."}]

