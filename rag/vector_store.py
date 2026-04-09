import uuid
import warnings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

try:
    from rag.chunker import create_chunks
except ImportError:
    from chunker import create_chunks

warnings.filterwarnings("ignore")


def main():
    client = QdrantClient(path="../qdrant_data")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    collection_name = "knowledge_base"

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    except Exception:
        pass

    texts = create_chunks()
    embeddings = embedding_model.encode(texts)

    points = []
    for text, embedding in zip(texts, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={"page_content": text},
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    client.close()


if __name__ == "__main__":
    main()
