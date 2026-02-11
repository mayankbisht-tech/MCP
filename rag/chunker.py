from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import load_all_pdfs

documents = load_all_pdfs()

def create_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    texts = [c.page_content for c in chunks]
    return texts

