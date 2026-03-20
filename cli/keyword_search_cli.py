#!/usr/bin/env python3

import argparse
import json

from lib.keyword_search import search_command as ks
from lib.search_utils import stem_word

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            stemmed_query = stem_word(args.query)
            results = ks(stemmed_query)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']}")
        case _:
            parser.print_help()
    

if __name__ == "__main__":
    main()
