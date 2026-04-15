import argparse
from semantic_search import verify_embedding, verify_model, embed_text


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Get term frequency for a document and term")
    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")
    subparsers.add_parser("verify_embeddings", help="Verify if our embeddings are correct")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embedding()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
