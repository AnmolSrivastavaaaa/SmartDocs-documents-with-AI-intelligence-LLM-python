# Private Knowledge Q&A

A web app to upload documents, ask questions, and get answers using AI.  
Answers are generated using a local GPT2 model and contextual embeddings via SentenceTransformers and ChromaDB.

## Features

- Upload text documents
- View list of uploaded documents
- Ask questions and get human-readable answers
- See which document the answer came from
- Simple and responsive UI with gradient design and animated buttons

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: HTML/CSS, Jinja2 templates
- **AI**: Local GPT2 model via Hugging Face Transformers
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector DB**: ChromaDB
- **Python packages**: fastapi, uvicorn, sentence-transformers, chromadb, transformers,python-dotenv

## Installation & Run

1. Clone the repo:
2. Install dependencies:
3. Run the app:
4. Open browser at `http://127.0.0.1:8000`

## What’s done
- Document upload & storage
- Embedding and chunking with SentenceTransformer
- GPT2 local answer generation
- Frontend with question input & answer display

## What’s not done
- Multi-user support
- Large-scale document search optimization

