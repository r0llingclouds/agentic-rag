import sys
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Initialize the config objects (these will load from .env)
from MilvusClient import MilvusClient
milvus_client = MilvusClient()

collection_name = os.getenv("MILVUS_COLLECTION")

if milvus_client.client.has_collection(collection_name):
    print(f"Collection {collection_name} already exists. Skipping.")
else:
    print(f"Collection {collection_name} does not exist. Creating...")
    job_positions_file_path = "job_positions.json"
    with open(job_positions_file_path, 'r') as f:
        job_positions = json.load(f)
    result = milvus_client.create_and_load_collection(collection_name, job_positions)

