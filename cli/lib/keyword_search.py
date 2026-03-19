import string
from turtle import title

from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movies, partial_search

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_no_punctuation = query.translate(str.maketrans("", "", string.punctuation)) # remove punctuation from query
        title_no_punctuation = movie["title"].translate(str.maketrans("", "", string.punctuation)) # remove punctuation from movie title
        
        if partial_search(query_no_punctuation, title_no_punctuation):
            results.append(movie)
            if len(results) >= limit:
                break
    return results
