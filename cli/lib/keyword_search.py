import string
from nltk.stem import PorterStemmer
import os
import pickle
from collections import defaultdict

from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords, CACHE_DIR


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """Search for movies whose titles partially match the query.

    Strips punctuation from both the query and movie titles before
    performing a partial token match. Returns up to ``limit`` results.

    Args:
        query (str): The search query.
        limit (int): Maximum number of results to return.
            Defaults to ``DEFAULT_SEARCH_LIMIT``.

    Returns:
        list[dict]: A list of matching movie records.
    """
    results = []

    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file is missing")
        return []
    
    query_tokens = tokenize_text(query)
    seen_id = set()
    if query_tokens:
        for query_token in query_tokens:
            doc_ids = idx.get_documents(query_token)

            for id in doc_ids:
                if id not in seen_id:
                    seen_id.add(id)
                    results.append(idx.docmap[id])
                    if len(results) >= limit:
                        return results
    return results

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text:str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    
    for token in tokens:
        if token:
            valid_tokens.append(token)
            
    stopwords = load_stopwords()
    filtered_words = []
    
    for word in valid_tokens:
        if word not in stopwords:
            filtered_words.append(word)
            
    stemmer = PorterStemmer()
    stemmed_words = []
    
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
        
    return list(set(stemmed_words))


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set) # maps each tokenized text to a set of docs_id
        self.docmap = {} # maps movies id to the movie object as per movie.json
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        for word in tokenize_text(text):
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)

    def get_documents(self, term):
        tokens = tokenize_text(term)
        if not tokens:
            return []
        token = tokens[0]
        doc_ids = self.index.get(token, set())
        return sorted(doc_ids)

    def build(self):
        movies = load_movies() 

        for movie in movies:
            self.docmap[movie["id"]] = movie

            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")

    def save(self):
        # when working with relative directory, we set file path based on current working directory which is where the script/process is run.
        # we run our script (cli/keyword_search_cli.py) from root directory, so that means "cache" is created in root directory.fffff
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self):
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)

            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("File does not exist")
