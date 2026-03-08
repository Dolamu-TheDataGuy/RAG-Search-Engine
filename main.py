import requests
import json


def main():
    print("Hello from rag-search-engine!")
    response = requests.get(
        "https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json"
    )
    movies = response.json()
    with open("data/movies.json", "w") as f:
        json.dump(movies, f, indent=4)

if __name__ == "__main__":
    main()
