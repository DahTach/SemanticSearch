import argparse
import os
from typing import List
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Document search using ChromaDB and LangChain"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="ChromaDB host address"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="ChromaDB port"
    )
    parser.add_argument(
        "--collection", type=str, default="documents", help="Collection name"
    )
    parser.add_argument(
        "--documents", type=str, help="Path to documents directory or file"
    )
    parser.add_argument(
        "--query", type=str, help="Search query"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000, help="Size of document chunks"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Overlap between chunks"
    )

    return parser.parse_args()

def load_documents(path: str) -> List:
    """Load documents from a file or directory."""
    try:
        print(f"Loading documents from: {path}")

        # Use UTF-8 encoding for TextLoader
        loader_kwargs = {"autodetect_encoding": True}

        if os.path.isdir(path):
            print("Loading files from directory matching pattern '**/*.md'")
            loader = DirectoryLoader(
                path,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs=loader_kwargs,
                show_progress=True
            )
            docs = loader.load()
            print(f"Loaded {len(docs)} documents from directory")
            return docs
        elif os.path.isfile(path):
            print("Loading single file")
            loader = TextLoader(path, **loader_kwargs)
            docs = loader.load()
            print(f"Loaded {len(docs)} documents from file")
            return docs
        else:
            raise ValueError(f"Invalid path: {path}")
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        raise
def create_chroma_db(docs, embedding_function, collection_name, client):
    """Create a ChromaDB database from documents."""
    if not docs:
        raise ValueError("No documents to add to the database. Please check your document loading process.")

    print(f"Creating ChromaDB with {len(docs)} documents")

    # Generate some sample embeddings to verify embedding function works
    try:
        sample_text = docs[0].page_content[:100] if docs[0].page_content else "Sample text for testing embeddings"
        sample_embedding = embedding_function.embed_query(sample_text)
        print(f"Sample embedding generated successfully. Length: {len(sample_embedding)}")
    except Exception as e:
        print(f"Error generating sample embedding: {str(e)}")
        raise

    # Load documents into Chroma
    try:
        db = Chroma.from_documents(
            docs,
            embedding_function,
            collection_name=collection_name,
            client=client
        )
        print(f"Added {len(docs)} document chunks to collection '{collection_name}'")
        return db
    except Exception as e:
        print(f"Error creating ChromaDB: {str(e)}")
        raise
    return db

def search_documents(db, query: str, k: int = 3):
    """Search documents in ChromaDB."""
    docs = db.similarity_search(query, k=k)

    print(f"\nSearch results for: '{query}'")
    print("-" * 50)

    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
        print("-" * 50)

    return docs

def main():
    args = parse_arguments()

    # Connect to ChromaDB (dockerized instance)
    client = chromadb.HttpClient(
        host=args.host,
        port=args.port,
        settings=Settings(allow_reset=True)
    )

    # Create embedding function
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Check if collection exists, or create it
    try:
        client.get_collection(args.collection)
        print(f"Connected to existing collection: {args.collection}")
    except:
        client.create_collection(args.collection)
        print(f"Created new collection: {args.collection}")

    # Load documents if path is provided
    if args.documents:
        try:
            documents = load_documents(args.documents)

            if not documents:
                print("No documents were loaded. Please check your files and try again.")
                return

            print(f"Successfully loaded {len(documents)} documents")

            # Print first few characters of first document for verification
            if documents and len(documents) > 0:
                print(f"Sample document content: {documents[0].page_content[:100]}...")

            # Split documents into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            docs = text_splitter.split_documents(documents)
            print(f"Split into {len(docs)} chunks")

            # Verify documents have content
            if not docs or all(not doc.page_content.strip() for doc in docs):
                print("Error: All document chunks are empty. Check your source files.")
                return

            # Create ChromaDB with the documents
            db = create_chroma_db(docs, embedding_function, args.collection, client)
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    else:
        # Connect to existing ChromaDB collection
        db = Chroma(
            client=client,
            collection_name=args.collection,
            embedding_function=embedding_function,
        )

    # Handle search query
    if args.query:
        search_documents(db, args.query)
    else:
        # Interactive search mode
        print("\nEnter your search query (or 'exit' to quit):")
        while True:
            query = input("> ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            if query.strip():
                search_documents(db, query)

if __name__ == "__main__":
    main()
