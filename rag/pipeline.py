import os
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from embedder import get_embeddings

COLLECTION = "knowledge_base"
MODEL_NAME = os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-base")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def ask(query, retriever):
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)[:1800]

    prompt = f"""Answer using the context below.
If the context is not enough, say that clearly.
Write at least 3 complete sentences.

Context:
{context}

Question: {query}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=140,
        min_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    client = None
    try:
        client = QdrantClient(path=str(PROJECT_ROOT / "qdrant_data"))
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION,
            embedding=get_embeddings(),
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    except RuntimeError as e:
        print(f"Qdrant is already in use by another process: {e}")
        raise SystemExit(1)

    try:
        while True:
            q = input("Ask: ").strip()
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            print(ask(q, retriever))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if client is not None:
            client.close()
