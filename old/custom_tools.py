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
from langchain_chroma import Chroma

# -------------------------------------------------------------------------
# 1) LOAD ENVIRONMENT VARIABLES
# -------------------------------------------------------------------------
load_dotenv(override=True)


elasticsearch_url = os.getenv("hostname_url", "") + ":" + os.getenv("port", "")
username = os.getenv("username", None)
password = os.getenv("password", None)

# For IBM WatsonX API:
api_key = os.getenv("WATSONX_API_KEY")
api_url = os.getenv("WATSONX_URL")
project_id = os.getenv("WATSONX_PROJECT_ID")
index_name = os.getenv("INDEX_NAME")

#os.environ["MEM0_API_KEY"] = "m0-oWbnC6Xz66tCBYhmZ5EBoO68hZZhGeEu4W8QUjiX"
# -------------------------------------------------------------------------
# 2) INIT OPENAI & ELASTICSEARCH CLIENTS
# -------------------------------------------------------------------------
es = Elasticsearch(
    elasticsearch_url,
    basic_auth=(username, password),
    max_retries=10,
    retry_on_timeout=True,
    verify_certs=False,  # Disable for production
    request_timeout=300
)

print("ES connection",es.ping())
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=api_key
)

params = TextChatParameters(
    temperature=0.7,
    max_tokens=2000
)

model = ModelInference(
    model_id="mistralai/mistral-large",
    credentials=credentials,
    project_id=project_id,
    params=params
)


embeddings = WatsonxEmbeddings(model_id='ibm/slate-125m-english-rtrvr',
                               apikey=credentials.get('apikey'),
                               url=credentials.get('url'),
                               project_id=project_id)

#### Custom DB tools Chroma
vector_store_chroma = Chroma(
    collection_name=index_name,
    embedding_function=embeddings,
    persist_directory="./data",  # Where to save data locally, remove if not necessary
)


# -------------------------------------------------------------------------
# 3) DEFINE PYDANTIC SCHEMAS & CLASS-BASED TOOLS
# -------------------------------------------------------------------------


class SearchInput(BaseModel):
    """Input schema for searching Elasticsearch with multi_match + text_expansion."""
    query: str = Field(..., description="User's search query.")
    top_n: int = Field(5, description="Number of top documents to return.")

class Chroma_Vectorstore_tool(BaseTool):
    """
    Creates a Custom Tool for Chroma DB data query
    """
    name: str = "search_chroma_tool"
    description: str = (
        "Searches 'ChromaDB' vector database to extract relevant documents from the index for RAG"
    )
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str, top_n: int=10) -> List[str]:
        """Retrieves course materials filtered by course name."""
        results = vector_store_chroma.similarity_search(query, k=top_n)
        return "\n\n".join([doc.page_content for doc in results])


class SearchElasticTool(BaseTool):
    """
    A tool to search using ELSER or multi_match queries.
    """
    name: str = "search_elastic_tool"
    description: str = (
        "Searches using a combination of multi_match "
        "and text_expansion for ELSER."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, top_n: int = 5) -> List[str]:
        """
        Execute a bool.should query:
          1) multi_match on 'text_field'
          2) text_expansion with ELSER

        Returns a list of text content from the matching documents.
        """
        # Build the query body
        body = {
            "sort": [
                {"_score": "desc"}
            ],
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "text_field"
                                ],
                                
                            }
                        },
                        {
                            "text_expansion": {
                                "ml.tokens": {
                                    "model_id": ".elser_model_2_linux-x86_64",
                                    "model_text": query,
                                    "boost": 0.5
                                }
                            }
                        }
                    ]
                }
            },
            "min_score": 5,
            "_source": ["text_field", "vector_query_field"],
            "size": top_n
        }

                

        # Send the request with a 60-second timeout
        print(index_name)
        response = es.search(
            index=index_name,
            body=body,
            request_timeout=60
        )

        # Extract 'text_field' from each hit
        return [
            hit["_source"].get("text_field", "No text_field found")
            for hit in response["hits"]["hits"]
        ]

