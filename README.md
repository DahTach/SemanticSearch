# Semantic Search

## Installation

```bash
uv sync
```

## Usage

```bash
# Run the demo script
demo [OPTIONS]

# Options:
#   --host TEXT         ChromaDB host address (default: localhost)
#   --port INTEGER      ChromaDB port (default: 8000)
#   --collection TEXT   Collection name (default: documents)
#   --documents TEXT    Path to documents directory or file
#   --query TEXT        Search query
#   --chunk-size INT    Size of document chunks (default: 1000)
#   --chunk-overlap INT Overlap between chunks (default: 200)

# Examples:

# Load documents from a directory into a collection
demo --documents ./samples --collection documents

# Load documents and run a single search query
demo --documents ./samples --query "your search query"

# Connect to an existing collection and run a search
demo --collection documents --query "your search query"

# Start interactive search mode with existing collection
demo --collection documents
```
