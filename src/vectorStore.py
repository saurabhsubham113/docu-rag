from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import requests
import os 

class VectorStore:
    def __init__(self,collection_name,embedding_model="text-embedding-3-large"):
        """Intializing vector store manager"""
        self.url = os.getenv("QDRANT_URL")
        self.collection_name = collection_name
        self.embedding_model = embedding_model
         # creating an open embedding function to embed the text
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)

    def get_existing_collections(self):
        """Connect an existing vector DB store collection"""
        retriever = QdrantVectorStore.from_existing_collection(
            collection_name=self.collection_name,
            url=self.url,
            embedding=self.embeddings

        )
        return retriever
    
    def create_vector_store(self,document=[]):
        QdrantVectorStore.from_documents(
            documents=document,
            collection_name = self.collection_name,
            url=self.url,
            embedding=self.embeddings
        )
        print("Injection done successfully")
    
    def collection_exists(self):
        """check if a collection exist in the vector DB"""
        try:
            response = requests.get(f"{self.url}/collections/{self.collection_name}")
            return response.status_code == 200
        except Exception as e:
            print(f" Error in checking collections {e}")
            return False
        
    def get_or_create_collection(self,document=[]):
        """get existing collection or create a new one"""
        if self.collection_exists():
            print(f" Collection already exists, Therefore using the same collections")
            return self.get_existing_collections()
        else:
            print(f"Collection {self.collection_name} does not exists,creating it with documents")
            if not document:
                ValueError("No documents provided for the collection")
            self.create_vector_store(document)
            return self.get_existing_collections()
