import argparse
from semantic_search import embed_query, verify_embedding, verify_model, embed_text, search_command, chunking_command


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Get term frequency for a document and term")
    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")
    subparsers.add_parser("verify_embeddings", help="Verify if our embeddings are correct")
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate embedding for a query")
    embed_query_parser.add_argument("query", type=str, help="Query to generate embedding for")
    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Maximum number of results to return")
    chunking_parser = subparsers.add_parser("chunk", help="Test chunking of documents")
    chunking_parser.add_argument("text", type=str, help="Text to chunk")
    chunking_parser.add_argument("--chunk-size", type=int, nargs="?", default=200, help="Size of each chunk")
    chunking_parser.add_argument("--overlap", type=int, nargs="?", default=20, help="Overlap between chunks")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embedding()
        case "embedquery":
            embed_query(args.query)
        case "search":
            result = search_command(args.query, args.limit)\
            
            count = 1
            for doc, score in result:
                print(f"{count}. {doc['title']} (Similarity: {score:.4f})\n {doc['description']}\n")
                count += 1
        case "chunk":
            chunks = chunking_command(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for idx, chunk in enumerate(chunks, start=1):
                print(f"{idx}. {chunk}\n")    
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
