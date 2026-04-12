import string
import math
from nltk.stem import PorterStemmer
import os
import pickle
from collections import defaultdict
from collections import Counter
from typing import Dict

from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords, CACHE_DIR, BM25_K1, BM25_B


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
    """
    Builds the inverted index from scratch and persists it to disk.
    """
    idx = InvertedIndex()
    idx.build()
    idx.save()


def tf_command(doc_id: int, term: str) -> int:
    """
    Loads the inverted index and returns the term frequency of a term in a document.

    Args:
        doc_id (int): Target document ID.
        term (str): Raw term to look up.

    Returns:
        int: Frequency count of the term in the document, or 0 if index is missing.
    """
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file is missing")
        return 0
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file is missing")
    return idx.get_idf(term)


def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file is missing in cache folder")
        return 0.0
    return idx.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file is missing in cache folder")
        return 0.0
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file is missing in cache folder")
        return 0.0
    return idx.get_bm25_tf(doc_id, term)

def bm25_search_command(query, limit=DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Index file is missing in cache folder")
        return []
    return idx.bm25_search(query, limit)


def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    """
    Checks if any query token is a substring match against any title token.

    Args:
        query_tokens (list[str]): Tokenized search query.
        title_tokens (list[str]): Tokenized document title.

    Returns:
        bool: True if at least one match is found, False otherwise.

    Example:
        >>> has_matching_tokens(["run", "fox"], ["fox", "jump"])
        True
    """
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    """
    Normalizes raw text by lowercasing and stripping punctuation.

    Args:
        text (str): Raw input string.

    Returns:
        str: Cleaned, lowercased text with punctuation removed.

    Example:
        >>> preprocess_text("Hello, World!")
        'hello world'
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    """
    Converts raw text into a list of stemmed tokens with stopwords removed.

    Pipeline: preprocess → split → filter empty → remove stopwords → stem

    Args:
        text (str): Raw input string to tokenize.

    Returns:
        list[str]: Stemmed tokens ready for RAG indexing or search.

    Example:
        >>> tokenize_text("The running foxes are quickly jumping")
        ['run', 'fox', 'quickli', 'jump']
    """
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

    return stemmed_words


class InvertedIndex:
    """
    Inverted index for keyword-based document retrieval.

    Maps tokenized terms to document IDs and tracks term frequencies
    per document. Supports persistent caching via pickle.

    Attributes:
        index (defaultdict[str, set]): Maps tokens to sets of doc IDs.
        docmap (dict): Maps doc IDs to movie objects from movies.json.
        term_frequencies (defaultdict[int, Counter]): Maps doc IDs to token frequency counters.
        index_path (str): Cache path for the index.
        docmap_path (str): Cache path for the docmap.
        term_frequencies_path (str): Cache path for term frequencies.
    """

    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Tokenizes and indexes a document's text against its ID.

        Args:
            doc_id (int): Unique document identifier.
            text (str): Raw text to tokenize and index.
        """
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for word in tokens:
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)
        term_counter = Counter(tokens)
        self.term_frequencies[doc_id] = term_counter


    def get_documents(self, term: str) -> list[int]:
        """
        Returns sorted doc IDs containing the given term.

        Args:
            term (str): Raw search term.

        Returns:
            list[int]: Sorted list of matching doc IDs, or empty list if no match.
        """
        tokens = tokenize_text(term)
        if not tokens:
            return []
        token = tokens[0]
        doc_ids = self.index.get(token, set())
        return sorted(doc_ids)


    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Returns the term frequency of a term within a document.

        Args:
            doc_id (int): Target document ID.
            term (str): Raw term to look up.

        Returns:
            int: Frequency count of the term in the document, or 0 if not found.
        """
        tokenized_term = tokenize_text(term)
        if len(tokenized_term) != 1:
            raise ValueError("Term must tokenize to exactly one token")

        term_counter = self.term_frequencies.get(doc_id, {})
        return term_counter.get(tokenized_term[0], 0) if tokenized_term else 0


    def get_idf(self, term: str) -> float:
        """
        Returns the inverse document frequency of a term across the corpus.

        Args:
            term (str): Raw term to look up.

        Returns:
            float: IDF score, or 0 if term is not found in any document.
        """
        tokenized_term = tokenize_text(term)
        if len(tokenized_term) != 1:
            raise ValueError("Term must tokenize to exactly one token")

        total_doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(tokenized_term[0], []))
        return math.log((total_doc_count + 1) / (term_doc_count + 1))


    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
   
   
    def get_bm25_idf(self, term: str) -> float:
        
        tokenize_term = tokenize_text(term)
        if len(tokenize_term) != 1:
            raise ValueError("Term must tokenize to exactly one token")
        
        total_doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(tokenize_term[0], []))
        return math.log((total_doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
    
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        length_norm = 1 - b + b * (self.doc_lengths.get(doc_id, 0) / self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf
    
    def bm25(self, doc_id: int, term: str):
        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query: str, limit: int):
        query_tokens = tokenize_text(query)
        doc_scores = {}
        for query_token in query_tokens:
            for doc_id in self.docmap:
                score = self.bm25(doc_id, query_token)
                if doc_id in doc_scores:
                    doc_scores[doc_id] += score
                else:
                    doc_scores[doc_id] = score 
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, self.docmap[doc_id], score) for doc_id, score in sorted_docs[:limit]] if sorted_docs else []


    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        total_length = sum(self.doc_lengths.values())
        avg_length = total_length / len(self.doc_lengths)
        return avg_length
 
    
    def build(self) -> None:
        """
        Loads movies and builds the index and docmap from scratch.
        """
        movies = load_movies()

        for movie in movies:
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")


    def save(self) -> None:
        """
        Persists the index, docmap, and term frequencies to disk as pickle files.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
            
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)


    def load(self) -> None:
        """
        Loads the index, docmap, and term frequencies from cached pickle files.

        Raises:
            FileNotFoundError: If any of the cache files are missing.
        """
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)

            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)

            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)

        except FileNotFoundError:
            raise FileNotFoundError("File does not exist")
