# Document QA Bot

A Retrieval Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its contents.

## Features

- Upload a PDF document
- Extract and chunk PDF text
- Generate semantic embeddings using Sentence Transformers
- Store and search embeddings using FAISS
- Retrieve relevant sections for a question
- Generate answers using the Groq API
- Display source page numbers used for the answer

## How It Works

```text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Sentence Transformer Embeddings
 ↓
FAISS Vector Search
 ↓
User Question
 ↓
Question Embedding
 ↓
Relevant Chunks
 ↓
Groq LLM
 ↓
Answer + Sources
```

## Tech Stack
- Python
- FastAPI
- PyMuPDF
- Sentence Transformers
- FAISS
- Groq API
- HTML
- CSS
- JavaScript

