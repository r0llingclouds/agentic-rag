from pymilvus import MilvusClient as MC
from pymilvus import AnnSearchRequest
from pymilvus import WeightedRanker, MilvusException, RRFRanker

from pymilvus import Function, FunctionType
from pymilvus import DataType
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

from Logger import Logger
from Singleton import Singleton


import os

class MilvusClient(metaclass=Singleton):
    """
    MilvusClient is a singleton class that provides a client for the Milvus database.
    It provides methods to create collections, insert data, and perform searches.
    """

    def __init__(self):

        self.logger = Logger('milvus_logger', os.getenv("LOG_MISC", "DEBUG")).logger

        self.client = MC(host="localhost", port="19530")

        self.logger.debug("Milvus Client successfully initialized.")

        self.embeddings = SentenceTransformerEmbeddingFunction("intfloat/e5-large-v2")
        self.sparse_embedding_function = None
        self.logger.debug("Dense embeddings initialized.")
        
    def create_schema(self, auto_id=False, enable_dynamic_field=True):
        """
        Create a schema for a Milvus collection with required fields and BM25 function.
        
        Args:
            auto_id (bool): Whether to auto-generate IDs
            enable_dynamic_field (bool): Whether to enable dynamic fields
        
        Returns:
            CollectionSchema: A complete schema object for the collection
        """
        
        # Create schema
        schema = self.client.create_schema(
            auto_id=auto_id,
            enable_dynamic_field=enable_dynamic_field,
        )


        # # Add fields to schema
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        
        # Job position specific fields
        schema.add_field(field_name="job_id", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="company_name", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="location", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="job_type", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="salary_range", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="required_skills", datatype=DataType.VARCHAR, max_length=500, enable_analyzer=True) # We will use this field for sparse search
        
        # Main text field for job description
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=5000)
        
        # Vector fields for search
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=self.embeddings.dim) 

        # Define function to generate sparse vectors
        bm25_function = Function(
            name="text_bm25_emb", # Function name
            input_field_names=["required_skills"], # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,
        )
        
        schema.add_function(bm25_function)
        
        return schema
    
    def create_indices(self, collection_name):
        """
        Prepare index parameters for both dense and sparse vector fields.
        
        Args:
            collection_name (str): Name of the collection for index preparation
        """
        # Prepare index parameters
        index_params = self.client.prepare_index_params()
        
        # Add indexes
        index_params.add_index(
            field_name="dense",
            index_name="dense_index",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        
        index_params.add_index(
            field_name="sparse",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",  # Index type for sparse vectors
            metric_type="BM25",  # Set to `BM25` when using function to generate sparse vectors
            params={"inverted_index_algo": "DAAT_MAXSCORE"},  # Algorithm for sparse index
        )
        
        return index_params
    
    def create_collection(self, collection_name, schema, index_params):
        """
        Create a new collection in Milvus with the provided schema and index parameters.
        
        Args:
            collection_name (str): Name of the collection
            schema (CollectionSchema): Schema for the collection
            index_params: Index parameters for the collection
        """
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        
        self.logger.debug(f"Created collection: {collection_name}")

    def prepare_data_for_insertion(self, documents):
        """
        Prepare documents for insertion into Milvus by generating dense embeddings.
        
        Args:
            documents (list): List of dictionaries with job position fields
            
        Returns:
            list: List of documents ready for insertion with dense embeddings
        """
        prepared_data = []
        
        # Extract text from documents
        texts = ["query: " + doc['text'] for doc in documents]
        
        # Generate dense embeddings in batch
        dense_vectors = self.embeddings(texts)
        
        for i, doc in enumerate(documents):
            # Format data for insertion
            data_point = {
                # auto_id=True so we don't need to provide the primary key (id)
                "job_id": doc.get('job_id', ''),
                "title": doc.get('title', ''),
                "company_name": doc.get('company_name', ''),
                "location": doc.get('location', ''),
                "job_type": doc.get('job_type', ''),
                "salary_range": doc.get('salary_range', ''),
                "required_skills": doc.get('required_skills', ''),
                "text": doc['text'],
                "dense": dense_vectors[i]
            }
            prepared_data.append(data_point)
        
        return prepared_data

    def insert_data(self, collection_name, prepared_data):
        """
        Insert data into a collection. Always prepares embeddings before insertion.
        
        Args:
            collection_name (str): Name of the collection
            data (list): List of dictionaries with 'id' and 'text' fields
            
        Returns:
            The result of the insert operation
        """

        self.logger.debug(f"Prepared {len(prepared_data)} documents with embeddings")
        
        # Insert prepared data
        res = self.client.insert(
            collection_name=collection_name,
            data=prepared_data
        )
        self.logger.debug(f"Inserted {len(prepared_data)} documents into {collection_name}")
        
        return res

    def create_and_load_collection(self, collection_name, documents):
        """
        Create a collection and load it with documents in one operation.
        
        Args:
            collection_name (str): Name of the collection
            documents (list): List of dictionaries with 'id' and 'text' fields
            
        Returns:
            dict: Result of the operation
        """
        # Check if collection exists
        if self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} already exists. Delete it before recreating it.")
        
        # Create schema and indices
        schema = self.create_schema()
        index_params = self.create_indices(collection_name)
        
        # Create collection
        self.create_collection(collection_name, schema, index_params)
        
        # Insert data
        prepared_data = self.prepare_data_for_insertion(documents)
        result = self.insert_data(collection_name, prepared_data)
        
        return result

    def list_collections(self) -> list:
        """
        List all collections in the Milvus instance.
        
        Returns:
            list: List of collection names
        
        Raises:
            MilvusException: If there was an error retrieving the collections
        """
        try:
            collections = self.client.list_collections()
            self.logger.debug(f"Found {len(collections)} collections")
            return collections
            
        except MilvusException as e:
            self.logger.error(f"Failed to list collections: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while listing collections: {str(e)}")
            raise

    def get_collection_stats(self, collection_name: str) -> dict:
        """
        Get detailed statistics about a collection including row count, fields and their types.
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            dict: Complete collection statistics and metadata
        """
        if not self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist")
            
        try:
            # Get row count from stats
            stats = self.client.get_collection_stats(collection_name)
            
            # Get full collection description
            description = self.client.describe_collection(collection_name)
            
            # Combine the information
            result = description.copy()
            result["row_count"] = stats.get("row_count", 0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            raise

    def delete_collection(self, collection_name):
        """
        Delete a collection from Milvus.
        """
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            self.logger.info(f"Collection {collection_name} deleted")
        else:
            self.logger.warning(f"Collection {collection_name} does not exist")
        
    def dense_search(self, collection_name, query_text, limit=5):
        """
        Perform a dense vector search using the query text.
        
        Args:
            collection_name (str): Name of the collection to search
            query_text (str): Text query to generate dense embedding
            limit (int, optional): Maximum number of results. Defaults to 10.
        
        Returns:
            list: List of search results with job position data
        """
        query_vector = self.embeddings(["query: " + query_text])[0]
        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="dense",
            limit=limit,
            output_fields=["job_id", "title", "company_name", "location", "job_type", 
                          "salary_range", "required_skills", "text"],
            search_params=search_params
        )[0]
        return results
    
    def sparse_search(self, collection_name, query_text, limit=5):
        """
        Perform a sparse vector search using the query text with BM25.
        
        Args:
            collection_name (str): Name of the collection to search
            query_text (str): Text query for sparse search
            limit (int, optional): Maximum number of results. Defaults to 10.
        
        Returns:
            list: List of search results with job position data
        """
        search_params = {"metric_type": "BM25", "params": {}}
        results = self.client.search(
            collection_name=collection_name,
            data=[query_text],
            anns_field="sparse",
            limit=limit,
            output_fields=["job_id", "title", "company_name", "location", "job_type", 
                          "salary_range", "required_skills", "text"],
            search_params=search_params
        )[0]
        return results

    def hybrid_search(self, collection_name, query_text, limit=5, ranker_type="weighted", **kwargs):
        """
        Perform a hybrid search combining dense and sparse vector searches.
        More info: https://milvus.io/docs/multi-vector-search.md
        
        Args:
            collection_name (str): Name of the collection to search
            query_text (str): Text query for generating dense embedding and sparse search
            ranker_type (str): Type of ranker to use ('weighted' or 'rrf')
            limit (int, optional): Maximum number of results. Defaults to 10.
            **kwargs: Parameters for the specific ranker:
                - If ranker_type is 'weighted': sparse_weight (default=0.3), dense_weight (default=0.7)
                - If ranker_type is 'rrf': k (default=60)
        
        Returns:
            list: List of search results with job position data
        """
        sparse_search_param = {
            "data": [query_text],
            "anns_field": "sparse",
            "param": {"metric_type": "BM25", "params": {}},
            "limit": limit
        }
        sparse_req = AnnSearchRequest(**sparse_search_param)

        query_dense_vector = self.embeddings(["query: " + query_text])[0]
        dense_search_param = {
            "data": [query_dense_vector],
            "anns_field": "dense",
            "param": {"metric_type": "IP", "params": {}},
            "limit": limit
        }
        dense_req = AnnSearchRequest(**dense_search_param)

        # Create appropriate ranker based on type
        if ranker_type.lower() == "weighted":
            sparse_weight = kwargs.get("sparse_weight", 0.3)
            dense_weight = kwargs.get("dense_weight", 0.7)
            ranker = WeightedRanker(sparse_weight, dense_weight)
            self.logger.debug(f"Using WeightedRanker with weights {sparse_weight} and {dense_weight}")
        elif ranker_type.lower() == "rrf":
            k = kwargs.get("k", 60)
            ranker = RRFRanker(k)
            self.logger.debug(f"Using RRFRanker with k={k}")
        else:
            raise ValueError(f"Unknown ranker type: {ranker_type}")

        results = self.client.hybrid_search(
            collection_name=collection_name,
            reqs=[sparse_req, dense_req],
            ranker=ranker,
            limit=limit,
            output_fields=["job_id", "title", "company_name", "location", "job_type", 
                          "salary_range", "required_skills", "text"]
        )[0]
        return results
