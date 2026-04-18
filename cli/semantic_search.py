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
    

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
    
        query_embedding = self.generate_embedding(query)
        similarities_store = []
    
        for idx, doc_embedding in enumerate(self.embeddings):
            similarities_store.append((self.documents[idx], cosine_similarity(query_embedding, doc_embedding)))
    
        # Sort by similarity (descending) and return top `limit` results
        similarities_store.sort(key=lambda x: x[1], reverse=True)
        return similarities_store[:limit]
    
    

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


def embed_query(query:str)-> None:
    semanticsearch = SemanticSearch()
    query_embedding = semanticsearch.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {query_embedding[:3]}")
    print(f"Shape: {query_embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def search_command(query, limit):
    semanticsearch = SemanticSearch()
    movies = load_movies()
    semanticsearch.load_or_create_embeddings(movies)
    return semanticsearch.search(query, limit)

def chunking_command(text, chunk_size, overlap):
    chunks = []
    splitted_text = text.split()
    
    # For loop approach with overlap
    # for i in range(0, len(splitted_text), chunk_size - overlap):
    #     chunk = splitted_text[i:i+chunk_size]
    #     chunks.append(" ".join(chunk))
    # return chunks
    
    # while loop approach with overlap : works better for small texts and edge cases
    start = 0
    while start+overlap < len(splitted_text):
        end = start + chunk_size
        chunk = splitted_text[start:end]
        chunks.append(" ".join(chunk))
        if overlap > 0:
            start += chunk_size - overlap
        else:
            start += chunk_size
    return chunks