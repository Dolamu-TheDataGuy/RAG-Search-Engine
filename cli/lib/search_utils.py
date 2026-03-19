import json
import os

DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
json_file_path = os.path.join(PROJECT_ROOT, "data", "movies.json")
stopwords_filepath = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_stopwords() -> list[str]:
    """Load and return a unique list of stopwords from a text file.

    Args:
        None

    Returns:
        list[str]: A deduplicated list of stopwords.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(stopwords_filepath, "r") as f:
        words = f.read().splitlines()
    return list(set(words))


def load_movies() -> list[dict]:
    """Load and return a list of movies from a JSON file.

    Args:
        None

    Returns:
        list[dict]: A list of movie records.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the JSON file lacks a top-level "movies" key.
    """
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return data["movies"]


def partial_search(query: str, title: str) -> bool:
    """Check if any query word partially matches any word in a movie title.

    Tokenizes and lowercases both the query and title, strips stopwords,
    then checks for substring matches between the two token sets.

    Args:
        query (str): The user's search query.
        title (str): The movie title to match against.

    Returns:
        bool: True if at least one query token is a substring of a title token,
            False otherwise.
    """
    split_query = [word for word in query.lower().split() if word]
    split_title = [word for word in title.lower().split() if word]

    stopwords = load_stopwords()
    split_query = [word for word in split_query if word not in stopwords]
    split_title = [word for word in split_title if word not in stopwords]

    for word_token in split_query:
        for title_token in split_title:
            if word_token in title_token:
                return True
    return False
