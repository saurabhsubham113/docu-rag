class DocumentRetriever:
    def __init__(self,vector_store):
        """Intialize retirever with a vector store"""
        self.vector_store = vector_store

    def retrieve_relevant_info(self,query,k=4):
        """Retreive the most relevant documents"""
        try:
            return self.vector_store.similarity_search(query=query,k=k)
        except Exception as e:
            print(f"error in retreiving documents {str(e)}")
            return []