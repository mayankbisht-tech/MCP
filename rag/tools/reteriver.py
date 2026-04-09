from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import groq
from dotenv import load_dotenv

from rag.model_config import GROQ_API_KEY, LLM_MODEL_NAME, LLM_PROVIDER

load_dotenv()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

COLLECTION_NAME = "knowledge_base"
QDRANT_PATH = str(Path(__file__).resolve().parents[2] / "qdrant_data")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = int(os.getenv("RAG_TOP_K", "2"))
MAX_DOC_CHARS = int(os.getenv("RAG_MAX_DOC_CHARS", "500"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "1000"))
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "64"))
MAX_GENERATION_TIME = float(os.getenv("RAG_MAX_GENERATION_TIME", "30"))
MIN_RELEVANCE_SCORE = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.45"))
SYSTEM_PROMPT = "You are a helpful AI assistant with strong knowledge of finance, trading, arbitrage, and bitcoin."


def load_components():
    from groq import Groq
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    client = QdrantClient(path=QDRANT_PATH)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    if LLM_PROVIDER.lower() != "groq":
        raise ValueError(
            f"Unsupported RAG_LLM_PROVIDER='{LLM_PROVIDER}'. Only 'groq' is configured."
        )
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY. Add it to your .env file or environment.")

    llm_client = Groq(api_key=GROQ_API_KEY, timeout=MAX_GENERATION_TIME, max_retries=0)
    return client, embedding_model, llm_client


def retrieve_documents(question, client, embedding_model, limit=TOP_K):
    query_vector = embedding_model.encode(question).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )

    documents = []
    for point in response.points:
        payload = point.payload or {}
        text = payload.get("page_content", "").strip()
        if not text:
            continue
        documents.append({"text": text, "score": getattr(point, "score", None)})
    return documents


def build_messages(question, documents):
    relevant_documents = [d for d in documents if isinstance(d.get("score"), (int, float)) and d["score"] >= MIN_RELEVANCE_SCORE]

    if not relevant_documents:
        return [
            {"role": "system", "content": f"{SYSTEM_PROMPT} Answer naturally and directly. If a question is not about finance, still help the user as a normal assistant. If you are unsure, say so clearly instead of making up facts."},
            {"role": "user", "content": question},
        ]

    context_blocks = [document["text"][:MAX_DOC_CHARS] for document in relevant_documents]
    context = "\n\n".join(context_blocks)[:MAX_CONTEXT_CHARS]

    return [
        {"role": "system", "content": f"{SYSTEM_PROMPT} Use the background information when it is relevant, but respond in a natural conversational way. Do not mention document numbers, retrieval, sources, or say things like 'the document says'. Give a clean answer as if you know the material. If the background is not enough, use your general knowledge carefully and say when you are uncertain."},
        {"role": "user", "content": f"User question:\n{question}\n\nHelpful background information:\n{context}\n\nAnswer the user directly in a natural way. Do not refer to the background material explicitly."},
    ]


def ask_bot(question, client, embedding_model, llm_client):
    retrieval_started = time.perf_counter()
    documents = retrieve_documents(question, client, embedding_model)
    retrieval_time = time.perf_counter() - retrieval_started

    messages = build_messages(question, documents)
    generation_started = time.perf_counter()

    try:
        completion = llm_client.chat.completions.create(
            messages=messages,
            model=LLM_MODEL_NAME,
            temperature=0,
            max_completion_tokens=MAX_NEW_TOKENS,
        )
    except groq.APITimeoutError:
        generation_time = time.perf_counter() - generation_started
        return (
            f"Generation timed out after {MAX_GENERATION_TIME:.0f} seconds.",
            retrieval_time,
            generation_time,
        )
    except groq.APIError as exc:
        generation_time = time.perf_counter() - generation_started
        return f"Groq API error: {exc}", retrieval_time, generation_time

    generation_time = time.perf_counter() - generation_started
    answer = (completion.choices[0].message.content or "").strip()
    if not answer:
        answer = "The model returned an empty response."
    return answer, retrieval_time, generation_time


def rag_chatbot():
    client = None
    try:
        client, embedding_model, llm_client = load_components()
    except Exception as exc:
        print(
            "Startup error: Qdrant local storage is locked by another running process. "
            "Close the other chatbot or Python session using '../qdrant_data' and try again."
            if "already accessed by another instance of Qdrant client" in str(exc)
            else f"Startup error: {exc}"
        )
        return

    try:
        while True:
            question = input("You: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            answer, retrieval_time, generation_time = ask_bot(
                question, client, embedding_model, llm_client
            )
            print(f"Bot: {answer}\n")
    except KeyboardInterrupt:
        print("\nExiting chatbot.")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    rag_chatbot()
