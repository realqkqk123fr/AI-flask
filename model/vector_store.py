from langchain.vectorstores import Chroma
from .embeddings import get_embeddings

def get_vector_store():
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory="gemini_data_100",
        embedding_function=embeddings
    )
    return vector_store
