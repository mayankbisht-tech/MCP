from langchain_community.document_loaders import PyPDFLoader
import os

def load_all_pdfs():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    loader1 = PyPDFLoader(os.path.join(parent_dir, "02Riskfree Arbitrage.pdf"), mode="single")
    loader2 = PyPDFLoader(os.path.join(parent_dir, "MSE444.pdf"), mode="single")
    loader3 = PyPDFLoader(os.path.join(parent_dir, "0137010028.pdf"), mode="single")

    docs1 = loader1.load()
    docs2 = loader2.load()
    docs3 = loader3.load()
    
    all_docs = docs1 + docs2 + docs3
    return all_docs
