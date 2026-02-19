import warnings
warnings.filterwarnings('ignore')

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from chunker import create_chunks
import uuid

client = QdrantClient(path="../qdrant_data")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

collection_name = "knowledge_base"
try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
except:
    pass

texts = create_chunks()
embeddings = embedding_model.encode(texts)

points = []
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={"page_content": text}
    ))

client.upsert(collection_name=collection_name, points=points)
print(f"Indexed {len(points)} documents")
client.close()
