import os

from dotenv import load_dotenv


load_dotenv()


LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "groq")
LLM_MODEL_NAME = os.getenv("RAG_LLM_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
