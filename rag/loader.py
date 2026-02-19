import os
from langchain_community.document_loaders import PyPDFLoader

def load_all_pdfs():
    folder_path = os.path.join(os.path.dirname(__file__), "books")
    
    all_docs = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    
    for pdf in pdf_files:
        full_path = os.path.join(folder_path, pdf)
        loader = PyPDFLoader(full_path, mode="single")
        all_docs.extend(loader.load())
    
    return all_docs

if __name__ == "__main__":
    load_all_pdfs()
