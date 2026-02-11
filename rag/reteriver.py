from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import transformers
import warnings

warnings.filterwarnings('ignore')

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(path="./qdrant_data")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="knowledge_base",
    embedding=embeddings
)

print("Vector DB Loaded")

pipe = transformers.pipeline(
    "text-generation",
    model="google/flan-t5-base",
    device="cpu",
    max_length=512,
    truncation=True
)

def ask_bot(question):
    docs = vector_store.similarity_search(question, k=3)

    if not docs:
        return "I don't know based on the documents."

    context = "\n\n".join([d.page_content for d in docs])
    
    if len(context) > 1500:
        context = context[:1500]

    prompt = f"""Answer the question using the context below.

Context:
{context}

Question: {question}

Answer:"""

    result = pipe(prompt, max_new_tokens=150, do_sample=False)

    return result[0]["generated_text"]

if __name__ == "__main__":
    print("RAG Chatbot Ready (type 'exit' to quit)\n")

    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break

        print("\nBot:", ask_bot(q), "\n")

    client.close()
