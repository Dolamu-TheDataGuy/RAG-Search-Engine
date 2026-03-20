import string
from turtle import title

from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movies, partial_search


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
    movies = load_movies()
    results = []
    for movie in movies:
        query_no_punctuation = query.translate(
            str.maketrans("", "", string.punctuation)
        )
        title_no_punctuation = movie["title"].translate(
            str.maketrans("", "", string.punctuation)
        )

        if partial_search(query_no_punctuation, title_no_punctuation):
            results.append(movie)
            if len(results) >= limit:
                break
    return results
