from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from chunker import create_chunks

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

COLLECTION = "knowledge_base"

texts = create_chunks()
documents = [Document(page_content=t) for t in texts]

vector_store = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    path="../qdrant_data",
    collection_name=COLLECTION
)

print("Documents successfully indexed")
