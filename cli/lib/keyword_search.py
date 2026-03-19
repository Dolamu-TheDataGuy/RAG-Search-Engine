from lib.search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_punctuation = query.translate(str.maketrans("", "", string.punctuation)) # remove punctuation from query
        movie_no_punctuation = movie['title'].translate(str.maketrans("", "", string.punctuation)) # remove punctuation from movie title
        if query_punctuation.lower() in movie_no_punctuation.lower():
            results.append(movie)
            if len(results) >= limit:
                break
    return results
