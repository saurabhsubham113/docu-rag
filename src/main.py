from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from documentProcessor import DocumentProcessor
from chatService import ChatService
from prompts import PromptGenerator
from vectorStore import VectorStore

def main():
    load_dotenv()
    pdf_path = Path(__file__).parent.parent / "public/nodejs.pdf"

    # Get the current file's directory and navigate to the root folder
    document_processor = DocumentProcessor(pdf_path)
    docs = document_processor.split_documents()
    
    chat_service = ChatService()
    vector_store = VectorStore('node_js')

    split_docs = docs["split_docs"]
    original_docs = docs["original_docs"]
    

    # vector_store.create_vector_store(split_docs)

    retriever = vector_store.get_or_create_collection(document=split_docs)

    while True:
        user_input = input("> ")

        if user_input.lower() in ['exit','quit','bye']:
            print("Good bye")
            break

        try:
            # get the releveant chunks from the vector DB
            relevant_chunks = retriever.similarity_search(
                query=user_input
            )

            # Generate system prompts
            sytem_prompt = PromptGenerator.create_system_prompts(relevant_chunks)

            response = chat_service.generate_response(sytem_prompt,user_input)

            print(f"\n {response} \n")
        except Exception as e:
            print(f" error occured in chatting {e}")
    

if __name__ == "__main__":
    main()