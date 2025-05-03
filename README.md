# DocuMind: Intelligent Document Query System

This project implements a Retrieval-Augmented Generation (RAG) system that allows you to ask questions about a PDF document and get accurate, context-aware responses.

## Overview

DocuMind provides an interactive command-line interface where users can query a PDF document's content. The system uses vector search technology to find relevant passages in the document and OpenAI's language models to generate human-like responses based on that context.

## Architecture

### High-Level System Flow

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│                  │     │                   │     │                  │
│   User Query     │────▶│ Document Retriever│────▶│   RAG Process    │
│                  │     │                   │     │                  │
└──────────────────┘     └───────────────────┘     └──────────┬───────┘
                                  ▲                           │
                                  │                           │
                         ┌────────┴───────┐                   │
                         │                │                   │
                         │  Vector Store  │                   │
                         │   (Qdrant)     │                   │
                         │                │                   ▼
                         └────────┬───────┘         ┌──────────────────┐
                                  │                 │                  │
                                  │                 │    Response      │
                                  │                 │                  │
                         ┌────────┴───────┐         └──────────────────┘
                         │                │
                         │    Document    │
                         │   Processor    │
                         │                │
                         └────────┬───────┘
                                  │
                                  │
                         ┌────────┴───────┐
                         │                │
                         │  PDF Document  │
                         │                │
                         └────────────────┘
```

### Detailed Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                        Document Processing Pipeline                     │
│                                                                         │
│  ┌───────────┐     ┌────────────────┐      ┌───────────────────┐        │
│  │           │     │                │      │                   │        │
│  │ PDF Loader│────▶│ Text Splitter  │─────▶│ Vector Embeddings │        │
│  │           │     │                │      │                   │        │
│  └───────────┘     └────────────────┘      └─────────┬─────────┘        │
│                                                      │                  │
└──────────────────────────────────────────────────────┼──────────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                           Retrieval System                              │
│                                                                         │
│  ┌──────────────┐      ┌─────────────────┐     ┌────────────────┐       │
│  │              │      │                 │     │                │       │
│  │ User Query   │─────▶│ Query Embedding │────▶│ Vector Search  │       │
│  │              │      │                 │     │                │       │
│  └──────────────┘      └─────────────────┘     └───────┬────────┘       │
│                                                        │                │
└────────────────────────────────────────────────────────┼────────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                          Generation Pipeline                            │
│                                                                         │
│  ┌────────────────┐     ┌─────────────────┐     ┌───────────────┐       │
│  │                │     │                 │     │               │       │
│  │ Context + Query│────▶│ Prompt Template │────▶│ LLM Response  │       │
│  │                │     │                 │     │               │       │
│  └────────────────┘     └─────────────────┘     └───────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### RAG System Data Flow

```
                        ┌────────────────────────────┐
                        │                            │
                        │   Document Pre-processing  │
                        │                            │
                        └───────────┬────────────────┘
                                    │
                                    ▼
                        ┌────────────────────────────┐
              ┌────────▶│                            │◀────────┐
              │         │   Vector Database (Qdrant) │         │
              │         │                            │         │
              │         └────────────────────────────┘         │
              │                                                │
┌─────────────┴────────────┐                      ┌────────────┴─────────┐
│                          │                      │                      │
│   OpenAI Embeddings API  │                      │   Document Chunks    │
│                          │                      │                      │
└─────────────┬────────────┘                      └────────────┬─────────┘
              │                                                │
              │                                                │
              ▼                                                ▼
      ┌─────────────────┐                          ┌─────────────────────┐
      │                 │                          │                     │
      │  Query Vector   │                          │   Content Vectors   │
      │                 │                          │                     │
      └────────┬────────┘                          └─────────┬───────────┘
               │                                             │
               │                                             │
               ▼                                             ▼
      ┌────────────────────────────────────────────────────────────────┐
      │                                                                │
      │               Semantic Similarity Matching                     │
      │                                                                │
      └────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │                     │
                        │  Top-K Relevant     │
                        │  Document Chunks    │
                        │                     │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │                     │
                        │   Prompt Template   │
                        │                     │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │                     │
                        │    OpenAI LLM API   │
                        │                     │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │                     │
                        │   Final Response    │
                        │                     │
                        └─────────────────────┘
```

## How RAG Works

Retrieval-Augmented Generation combines the power of retrieval-based systems with generative AI:

### 1. Document Processing

The system takes a PDF document and breaks it down into smaller, manageable chunks with strategic overlap to maintain context across sections. This is handled by the `DocumentProcessor` class using the `RecursiveCharacterTextSplitter` which intelligently splits text at meaningful boundaries.

```python
# Example of how chunking works
Original text: "Node.js is a JavaScript runtime built on Chrome's V8 JavaScript engine. Node.js uses an event-driven, non-blocking I/O model that makes it lightweight and efficient."

Chunk 1: "Node.js is a JavaScript runtime built on Chrome's V8 JavaScript engine."
Chunk 2: "...JavaScript engine. Node.js uses an event-driven, non-blocking I/O model..."
Chunk 3: "...non-blocking I/O model that makes it lightweight and efficient."
```

### 2. Vectorization

Each text chunk is converted into a high-dimensional vector (embedding) using OpenAI's embedding models. These embeddings capture the semantic meaning of the text in a way that computers can process.

```
Text: "Node.js is a JavaScript runtime"
↓ [OpenAI Embedding]
Vector: [0.021, -0.043, 0.012, 0.036, ..., 0.019] (1536 dimensions)
```

### 3. Storage

The vector embeddings are stored in Qdrant, a specialized vector database that enables efficient similarity search. Each vector is indexed for quick retrieval, making it possible to quickly find relevant document chunks.

### 4. Retrieval

When a user asks a question, the system:

- Converts the question into a vector embedding using the same embedding model
- Performs a "similarity search" in Qdrant to find document chunks with the closest semantic meaning
- Retrieves the top-k most relevant chunks (by default, k=4)

### 5. Generation

The retrieved text chunks are combined with the user's query in a carefully designed prompt template that instructs the AI on how to use the provided context. This enhanced prompt is sent to OpenAI's language model.

### 6. Response

The language model generates a response that's grounded in the specific context from the original document, providing accurate and relevant answers with proper attribution to the source material.

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- OpenAI API key

## Installation

1. Clone the repository:

   ```
   git clone [repository-url]
   cd web-rag
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key:

   ```
   OPENAI_API_KEY=your_api_key_here
   QDRANT_URL=http://localhost:6333
   ```

5. Start the Qdrant vector database:
   ```
   docker-compose up -d
   ```

## Usage

1. Place your PDF document in the `public/` directory (the default is set to `nodejs.pdf`)

2. Run the main application:

   ```
   python src/main.py
   ```

3. Start asking questions about your document at the prompt:

   ```
   > What are the core features of Node.js?
   ```

4. Type `exit`, `quit`, or `bye` to end the session.

## Project Components

### DocumentProcessor

Responsible for loading and splitting PDF documents into smaller, manageable chunks:

```python
# Key functions:
load_pdf() # Loads a PDF from file path
split_documents() # Splits the document into chunks with overlap
```

### VectorStore

Manages interactions with the Qdrant vector database:

```python
# Key functions:
create_vector_store() # Creates a new collection with document vectors
get_existing_collections() # Connects to an existing collection
collection_exists() # Checks if a collection already exists
get_or_create_collection() # Gets or creates a collection as needed
```

### DocumentRetriever

Retrieves relevant document chunks based on semantic similarity to the query:

```python
# Key functions:
retrieve_relevant_info() # Finds semantically similar document chunks
```

### ChatService

Handles interactions with OpenAI's language models:

```python
# Key functions:
generate_response() # Creates AI response with given context and query
clear_chat_history() # Resets the conversation history
```

### PromptGenerator

Creates system prompts with relevant context for high-quality, accurate responses:

```python
# Key functions:
create_system_prompts() # Builds a prompt template with retrieved context
```

## Customization

- **Document Processing**: Modify the `chunk_size` and `chunk_overlap` in `DocumentProcessor` for different document types. Longer chunks provide more context but may decrease retrieval precision.

  ```python
  # For technical documents with complex content:
  document_processor = DocumentProcessor(pdf_path, chunk_size=1500, chunk_overlap=300)

  # For simpler documents:
  document_processor = DocumentProcessor(pdf_path, chunk_size=800, chunk_overlap=150)
  ```

- **Language Model**: Change the language model in `ChatService` constructor:

  ```python
  # For high accuracy (default):
  chat_service = ChatService(model="gpt-4.1")

  # For faster responses or lower cost:
  chat_service = ChatService(model="gpt-3.5-turbo")
  ```

- **Retrieval Parameters**: Adjust the number of retrieved chunks with the `k` parameter in `retrieve_relevant_info`:

  ```python
  # More context (may include less relevant info):
  relevant_chunks = retriever.retrieve_relevant_info(query=user_input, k=6)

  # More focused context:
  relevant_chunks = retriever.retrieve_relevant_info(query=user_input, k=3)
  ```

## Troubleshooting

- **Connection Errors with Qdrant**:

  - Ensure the Docker container is running: `docker ps | grep qdrant`
  - Check Docker logs: `docker logs qdrant`
  - Verify the QDRANT_URL in your .env file matches the running container

- **OpenAI-related Errors**:

  - Verify your API key is correctly set in the .env file
  - Check for rate limiting or usage limits in your OpenAI account
  - If you get model-related errors, ensure you're using a valid model name

- **Document Retrieval Issues**:

  - If retrieval seems inaccurate, try:
    - Adjusting chunk sizes (smaller for more precise retrieval)
    - Increasing chunk overlap for better context preservation
    - Using a more powerful embedding model (e.g., "text-embedding-3-large")

- **PDF Loading Problems**:
  - Some PDFs may be encrypted or have other access restrictions
  - Try converting the PDF to a different format or using a different PDF

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
