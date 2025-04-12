### Example data ingestion
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM, ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenTextParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
from langchain_elasticsearch import ElasticsearchStore
import os
from ibm_watsonx_ai import Credentials


load_dotenv(override=True)
api_key = os.getenv("WATSONX_API_KEY")
api_url = os.getenv("WATSONX_URL")
project_id = os.getenv("WATSONX_PROJECT_ID")
index_name=os.getenv("INDEX_NAME")
# Elasticsearch Configuration
ELASTICSEARCH_HOST = os.getenv("hostname_url", "http://localhost:9200")+ ':' + os.getenv("port")
ELASTICSEARCH_USER = os.getenv("username")
ELASTICSEARCH_PASSWORD = os.getenv("password")

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=api_key
)

embeddings = WatsonxEmbeddings(model_id='ibm/slate-125m-english-rtrvr',
                               apikey=credentials.get('apikey'),
                               url=credentials.get('url'),
                               project_id=project_id)

elser_index_mapping={
    "mappings": {
        "properties": {
            "ml.tokens": {"type": "rank_features"},
            "content": {"type": "text"},
        }
    }
}

INDEX_SETTINGS = {
    "index": {
        "number_of_replicas": "1",
        "number_of_shards": "1",
        "default_pipeline": "agentic_rag",
    }
}

# Initialize Elasticsearch Client
es = Elasticsearch(
    [ELASTICSEARCH_HOST],
    basic_auth=(ELASTICSEARCH_USER, ELASTICSEARCH_PASSWORD),
    verify_certs=False,
    timeout=30
)


def chroma_db_ingestion():
    urls = [
        "https://www.ibm.com/products/watsonx-ai",
        "https://www.ibm.com/products/watsonx-orchestrate?lnk=flatitem",
        "https://www.ibm.com/products/watsonx-assistant?lnk=flatitem"
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)

    try:
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=index_name,
            embedding=embeddings,
            persist_directory='./data'
        )
        print("Data loaded")
    except Exception as e:
        print("Error:", e)


def elastic_ingestion():
    """
    Fetch documents, split them into chunks, and store them in Elasticsearch using ELSER.
    """
    urls = [
        "https://www.ibm.com/products/watsonx-ai",
        "https://www.ibm.com/products/watsonx-orchestrate?lnk=flatitem",
        "https://www.ibm.com/products/watsonx-assistant?lnk=flatitem"
    ]

    # Load documents from URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Prepare metadata & content for text splitting
    metadata = []
    content = []

    for doc in docs_list:
        content.append(doc.page_content)
        metadata.append(
            {
                "source": doc.metadata.get("source", ""),
                "title": doc.metadata.get("title", ""),
            }
        )

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=256
    )
    doc_splits = text_splitter.create_documents(content, metadatas=metadata)
    
    vector_store = ElasticsearchStore(
        es_connection=es,
        index_name=index_name,
        query_field="text_field",
        vector_query_field="ml",
        strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(
                model_id=".elser_model_2_linux-x86_64"  # ELSER Model
            )
    )

    if vector_store.client.indices.exists(index=index_name):
            vector_store.client.indices.delete(index=index_name, ignore=[400, 404])
    vector_store.client.indices.create(
        index=index_name, mappings=elser_index_mapping, settings=INDEX_SETTINGS, ignore=[400, 404]
    )

    # Store in Elasticsearch using ELSER model
    try:
        vector_store = ElasticsearchStore.from_documents(
            doc_splits,
            index_name=index_name,
            es_connection=es,
            query_field="text_field",
            vector_query_field="ml",
            strategy=ElasticsearchStore.SparseVectorRetrievalStrategy(
                model_id=".elser_model_2_linux-x86_64"  # ELSER Model
            ),
            bulk_kwargs={"request_timeout": 60},
        )
        print(f"Successfully ingested {len(doc_splits)} documents into Elasticsearch.")
    except Exception as e:
        print("Elasticsearch Ingestion Error:", e)

if __name__ == "__main__":
    chroma_db_ingestion()