import os
from typing import Type, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from ibm_watsonx_ai.foundation_models import ModelInference
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai import LLM
from elasticsearch import Elasticsearch
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai import Credentials
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from langchain_ibm import WatsonxEmbeddings
from chromadb.utils.embedding_functions import create_langchain_embedding
from custom_tools import Chroma_Vectorstore_tool, SearchElasticTool
#from mem0 import MemoryClient

# -------------------------------------------------------------------------
# 1) INSTANTIATE TOOL OBJECTS
# -------------------------------------------------------------------------
search_elastic_tool = SearchElasticTool()
chroma_vectorstore_tool = Chroma_Vectorstore_tool()

# -------------------------------------------------------------------------
# 2) DEFINE AGENTS
# -------------------------------------------------------------------------
retrieval_agent = Agent(
    role="Vector Database Retrieval Specialist",
    goal="Retrieve relevant documents from ELasticsearch based on the provided {query}"
        "Modify the given {query} retaining the scemantic meaning of the {query} if the results from search_elastic_tool are empty.",
    backstory="AI specialized in investment search using Elasticsearch.",
    verbose=True,
    tools=[chroma_vectorstore_tool],
    llm="watsonx/meta-llama/llama-3-3-70b-instruct",
    max_iter=5,
    #memory=True
)


answer_agent = Agent(
    role="Answer Agent",
    goal="Analyze retrieved documents to answer the {query}",
    backstory="Expert AI assistant providing user-friendly responses.",
    verbose=True,
    #memory=True,
    tools=[],  # Not needed, as it formats responses
    llm="watsonx/meta-llama/llama-3-405b-instruct"
)

Grader_agent =  Agent(
  role='Documents Grader',
  goal='Filter out erroneous retrievals',
  backstory=(
    "You are a grader assessing relevance of a retrieved document to a user query."
    "If the document contains keywords related to the user query, grade it as relevant."
    "It does not need to be a stringent test.You have to make sure that the answer is relevant to the query."
  ),
  verbose=True,
  allow_delegation=False,
  llm="watsonx/meta-llama/llama-3-3-70b-instruct",
)

hallucination_grader = Agent(
    role="Hallucination Grader",
    goal="Filter out hallucination",
    backstory=(
        "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts."
        "Make sure you meticulously review the answer and check if the response provided is in alignmnet with the question asked"
    ),
    verbose=True,
    allow_delegation=False,
    llm="watsonx/meta-llama/llama-3-3-70b-instruct")

answer_grader = Agent(
    role="Answer Grader",
    goal="Filter out hallucination from the answer.",
    backstory=(
        "You are a grader assessing whether an answer is useful to resolve a question."
        "Make sure you meticulously review the answer and check if it makes sense for the question asked"
        "If the answer is relevant from answer_task return the response from answer_task."
        "If the answer gnerated is not relevant then Otherwise respond as 'Sorry! unable to find a valid response'."
    ),
    verbose=True,
    allow_delegation=False,
    llm="watsonx/meta-llama/llama-3-3-70b-instruct",
    max_iter=1,
)


# -------------------------------------------------------------------------
# 3) DEFINE TASKS
# -------------------------------------------------------------------------

retrieval_task = Task(
    description="Retrieve relevant documentsfrom Elasticsearch.",
    expected_output="A list of relevant documents from Elasticsearch",
    agent=retrieval_agent
)



grader_task = Task(
    description=("Based on the response from the retriever task for the question {query} evaluate whether the retrieved content is relevant to the question."
    ),
    expected_output=("Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
    "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
    "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
    "You must answer 'no' if the response from the 'retriever_task' is [] or an empty string."
    "Do not provide any preamble or explanations except for 'yes' or 'no'."),
    agent=Grader_agent,
    context=[retrieval_task],
)

answer_task = Task(
    description="Based on the {query} provide response if grader_task output is 'yes', Add bullet points where needed, Extract key information w.r.t Company Names, Sector, Geography, Key contacts, web links etc, Respond based on only the output of retrieval_task, Donot Add any verbose reasoning and donot mention about the retreiver source."
    "If the grader_task output is 'no' then repsond as 'Sorry! unable to answer, Please ask another question.'",
    expected_output="A structured response detailing investment opportunities.",
    agent=answer_agent,
    context=[retrieval_task, grader_task]
)

hallucination_task = Task(
    description=("Based on the response from the answer_task for the question {query} evaluate whether the answer from answer_task is grounded in / supported by a set of facts."),
    expected_output=("Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked"
    "Respond 'yes' if the answer is in useful and contains fact about the question asked. and grader_task output is 'yes'"
    "Respond 'no' if the answer is not useful and does not contains fact about the question asked and grader_task output is 'no'"
    "Do not provide any preamble or explanations except for 'yes' or 'no'."),
    agent=hallucination_grader,
    context=[answer_task, grader_task],
)

answer_grader_task = Task(
    description=("Based on the response from the hallucination task for the question {query} evaluate whether the answer generated from the answer_task is useful to resolve the question."
    "If the answer is 'yes' return the response from the answer_task."
    "If the answer is 'no' then respond as 'Sorry! unable to answer, Please ask another question.'"),
    expected_output=("Return the response of answer_task if the response from 'hallucination_task' is 'yes' and the answer generated from the answer_task is useful to resolve the question."
    "Otherwise respond as 'Sorry! unable to answer, Please ask another question'."
    "Donot mention about the hallucination task in your response, you are expected to pass the answer from answer_task if hallucination_task reposne is 'yes'"),
    context=[hallucination_task, answer_task],
    agent=answer_grader,
    #tools=[answer_grader_tool],
)
# -------------------------------------------------------------------------
# 4) CREATE THE CREW
# -------------------------------------------------------------------------
crew_chain = Crew(
    agents=[retrieval_agent, Grader_agent,answer_agent, hallucination_grader, answer_grader],
    tasks=[retrieval_task, grader_task,answer_task, hallucination_task, answer_grader_task],
    #memory=True,
    #embedder=wx_chroma_embedder,
    process=Process.sequential
)
