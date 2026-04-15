import numpy as np
import os

from sentence_transformers import SentenceTransformer
from lib.search_utils import load_movies

class SemanticSearch:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text:str):
        if len(text) == 0 or text.isspace():
            raise ValueError("Input text cannot be empty or whitespace.")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents):
        self.documents = documents
        doc_list = []
        for document in documents:
            self.document_map[document["id"]] = document
            doc_list.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)

        if not os.path.exists("cache"):
            os.mkdir("cache")
        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            print("Loaded embeddings from cache.")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            print("No cached embeddings found. Building new embeddings.")
            self.build_embeddings(documents)


def verify_model():
    semantic_search = SemanticSearch()
    
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str):
    semanticsearch = SemanticSearch()
    embedding = semanticsearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embedding():
    semanticsearch = SemanticSearch()
    
    movies = load_movies()
    semanticsearch.load_or_create_embeddings(movies)
    print(f"Number of docs: {len(semanticsearch.documents)}")
    print(f"Embedding shape: {semanticsearch.embeddings.shape[0]} vectors in {semanticsearch.embeddings.shape[1]} dimensions")
    