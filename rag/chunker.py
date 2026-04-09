from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

try:
    from rag.loader import load_all_pdfs
except ImportError:
    from loader import load_all_pdfs

def create_chunks():
    documents = load_all_pdfs()
    
    cleaned_docs = []
    for doc in documents:
        text = doc.page_content
        text = re.sub(r'©\d{4}.*?Inc\.', '', text, flags=re.DOTALL)
        text = re.sub(r'Vice President.*?Manufacturing Buyer.*?Uhrig', '', text, flags=re.DOTALL)
        text = re.sub(r'This page intentionally left blank', '', text)
        text = re.sub(r'\n+', '\n', text)
        
        if len(text.strip()) > 50:
            doc.page_content = text.strip()
            cleaned_docs.append(doc)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(cleaned_docs)
    return [c.page_content for c in chunks if len(c.page_content) > 100]
