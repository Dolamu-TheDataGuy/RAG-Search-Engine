import json
import os

DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
json_file_path = os.path.join(PROJECT_ROOT, "data", "movies.json")
stopwords_filepath = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")



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
