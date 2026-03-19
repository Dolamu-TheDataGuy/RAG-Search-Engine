import json
import os

DEFAULT_SEARCH_LIMIT = 5
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
json_file_path = os.path.join(PROJECT_ROOT, "data", "movies.json")



def load_movies() -> list[dict]:
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return data["movies"]
        
