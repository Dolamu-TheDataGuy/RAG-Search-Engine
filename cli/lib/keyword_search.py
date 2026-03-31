import string
from nltk.stem import PorterStemmer
import os
import pickle
from collections import defaultdict
from collections import Counter

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

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Tokenizes and indexes a document's text against its ID.

        Args:
            doc_id (int): Unique document identifier.
            text (str): Raw text to tokenize and index.
        """
        tokens = tokenize_text(text)
        for word in tokens:
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)

        term_counter = Counter(tokens)
        self.term_frequencies[doc_id] = term_counter

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
        term_counter = self.term_frequencies.get(doc_id, {})
        return term_counter.get(tokenized_term[0], 0) if tokenized_term else 0

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

        except FileNotFoundError:
            raise FileNotFoundError("File does not exist")
