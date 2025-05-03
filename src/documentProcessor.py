from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self,pdf_path=None,chunk_size=1000,chunk_overlap=200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self,pdf_path=None):
        """Load pdf from the path provided"""
        if pdf_path:
            self.pdf_path = pdf_path
        
        if not self.pdf_path:
            raise ValueError("Pdf path not specified")
        
        loader = PyPDFLoader(file_path=self.pdf_path)
        return loader.load()

    def split_documents(self):
        """split docs into chunks based on the config provided"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
        )
        original_docs = self.load_pdf(self.pdf_path)

        split_docs =  text_splitter.split_documents(documents=original_docs)
        return {
            "original_docs":original_docs,
            "split_docs":split_docs
        }