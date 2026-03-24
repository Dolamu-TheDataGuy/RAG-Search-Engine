#!/usr/bin/env python3

import argparse
import json

from lib.keyword_search import search_command, tokenize_text

def main() -> None:
    """Entry point for the Keyword Search CLI.

    Parses command-line arguments and dispatches to the appropriate command.
    Currently supports the following command:

    - ``search <query>``: Stems the query and retrieves ranked movie results via BM25.

    Example:
        .. code-block:: bash

            $ python app.py search "inception"
            1. Inception
            2. The Dark Inception
    """
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']}")
        case _:
            parser.print_help()


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        
    def __add_document(self, doc_id: int, text: str) -> None:
        for word in tokenize_text(text):
            if word not in self.index:
                self.index[word] = set()
            self.index[word].add(doc_id)



if __name__ == "__main__":
    main()
