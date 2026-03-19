import json
import os
import string

DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
json_file_path = os.path.join(PROJECT_ROOT, "data", "movies.json")



def load_movies() -> list[dict]:
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return data["movies"]
        
def partial_search(query, title):
    split_query = query.lower().split()
    split_query = [word for word in split_query if word] # remove empty strings from split query
    split_title = title.lower().split()
    split_title = [word for word in split_title if word] # remove empty strings from split title
    
    for word in split_query:
        for token in split_title:
            if word in token:
                return True
    return False
