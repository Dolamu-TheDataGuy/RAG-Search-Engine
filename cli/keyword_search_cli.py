#!/usr/bin/env python3

import argparse
import math

from lib.keyword_search import search_command, build_command, tf_command, InvertedIndex
from lib.search_utils import load_movies


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
    subparsers.add_parser("build", help="Build Inverted index algorithm")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")
    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']}")
        case "build":
            print("Building Inverted index....")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(tf)
        case "idf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError:
                print("Index file is missing")
            total_doc_count = len(idx.docmap)
            total_match_doc_count = len(idx.get_documents(args.term))
            
            idf = math.log((total_doc_count + 1)/ (total_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
