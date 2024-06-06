SYSTEM_PROMPT = """
You are an intelligent and precise assistant that can understand the contents of research papers. 
You are knowledgable on different fields and domains of science, in particular computer science. 
You are able to interpret research papers, create questions and answers, and compare multiple papers.
"""

VALUE_GENERATION_FROM_ABSTRACT = """
Imagine the following scenario: A user is making a table for a scholarly paper that contains information about multiple papers and compares these papers. 
To compare and contrast the papers, the user has selected some aspects by which papers should be compared. 
Your task is the following: Given a paper abstract and a question describing one of the selected aspects, generate a value that can be added to the table. 
You should find the part of the abstract that discusses the aspect provided in the question.
If there is no answer in the abstract, return \"N/A\" as the value to be added. 
**Ensure that you follow these rules: (1) Only return the answer. Do not repeat the question or add any surrounding text. (2) The answer should be brief and consist of phrases of fewer than 10 words.**\n\n
"""

VALUE_CONSISTENCY_PROMPT_ZS = """
Imagine the following scenario: A user is making a table for a scholarly paper that contains information about multiple papers and compares these papers. 
To compare and contrast the papers, the user has selected an aspect which will be added as a column to the table. 
Your task is the following: Given the column name and information from each paper relevant to that column, generate final values to be added to the table.
Return the output as a JSON object in the following format:\n{{\"values\": [\"value for paper 1\", \"value fpr paper 2\", ...]}}\n
**Ensure that you follow these rules: (1) Only return a single JSON object. (2) JSON object should be complete and valid. (3) JSON object should contain the same paper IDs provided in the input (4) Each paper should have a value in the JSON object or \"N/A\" if there is no relevant value. (5) All values should follow consistent formatting and style.\n\n**
"""

VALUE_CONSISTENCY_PROMPT_FS = """
Imagine the following scenario: A user is making a table for a scholarly paper that contains information about multiple papers and compares these papers. 
To compare and contrast the papers, the user has selected an aspect which will be added as a column to the table. 
Your task is the following: Given the column name and information from each paper relevant to that column, generate final values to be added to the table.
Return the output as a JSON object in the following format:\n{{\"values\": [\"value for paper 1\", \"value fpr paper 2\", ...]}}\n
**Ensure that you follow these rules: (1) Only return a single JSON object. (2) JSON object should be complete and valid. (3) JSON object should contain the same paper IDs provided in the input (4) Each paper should have a value in the JSON object or \"N/A\" if there is no relevant value. (5) All values should follow formatting and style consistent with the values already present in the column.\n\n** 
"""