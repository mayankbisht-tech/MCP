import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np

def rag_chatbot():
    try:
        client = QdrantClient(path="./qdrant_data")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error: {e}")
        return
    
    def ask_bot(question):
        query_vector = embedding_model.encode([question])[0].tolist()
        
        try:
            search_results = client.search(
                collection_name="knowledge_base",
                query_vector=query_vector,
                limit=3
            )
        except AttributeError:
            search_results = client.query_points(
                collection_name="knowledge_base",
                query=query_vector,
                limit=3
            ).points
        
        if not search_results:
            return "No relevant information found."
        
        contexts = []
        for result in search_results:
            if hasattr(result, 'score') and result.score > 0.5:
                contexts.append(result.payload.get('page_content', ''))
            else:
                contexts.append(result.payload.get('page_content', ''))
        
        if not contexts:
            return "No sufficiently relevant information found."
        
        return contexts[0][:800] + "..." if len(contexts[0]) > 800 else contexts[0]
    
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break
        answer = ask_bot(q)
        print(f"Bot: {answer}\n")
    
    client.close()

if __name__ == "__main__":
    rag_chatbot()
