# pdf_loader.py

from langchain_community.document_loaders import PyPDFLoader

def load_all_pdfs():
    loader1 = PyPDFLoader("02Riskfree Arbitrage.pdf", mode="single")
    loader2 = PyPDFLoader("MSE444.pdf", mode="single")
    loader3 = PyPDFLoader("0137010028.pdf", mode="single")

    docs1 = loader1.load()
    docs2 = loader2.load()
    docs3 = loader3.load()

    
    all_docs = docs1 + docs2 + docs3

    return all_docs
