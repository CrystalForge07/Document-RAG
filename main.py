
import pickle, numpy as np, faiss, os
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pymupdf

app = FastAPI()

@app.get("/")
def home():
    return FileResponse("frontend/index.html")

# Load saved data

chunks = []
faiss_index = None

# Load models

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Retrieval (and augmentation)

def retrieve(question):

    question_embedding = embedding_model.encode(
        [question]
    )

    top_k = 5

    distances, indices = faiss_index.search(
        question_embedding,
        top_k
    )

    context = "\n\n".join(
        chunks[chunk_index]
        for chunk_index in indices[0]
    )

    return context

def load_and_chunk_pdf(pdf_path):

    pdf = pymupdf.open(pdf_path)

    chunks = []

    for page in pdf:

        text = page.get_text()
        words = text.split()

        chunk_size = 200
        overlap = 40

        for i in range(0, len(words), chunk_size - overlap):

            chunk = " ".join(
                words[i:i + chunk_size]
            )

            chunks.append(chunk)

    pdf.close()

    return chunks

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    global chunks, faiss_index

    with open("uploaded.pdf", "wb") as buffer:
        buffer.write(file.file.read())

    chunks = load_and_chunk_pdf("uploaded.pdf")

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(dimension)

    faiss_index.add(embeddings)

    return {
        "message": "PDF uploaded successfully!",
        "chunks": len(chunks)
    }

# Generation

def generate_answer(question, context):

    prompt = f"""
Answer the question using only the provided context.

For each part of the question:
- If the information is present in the context, provide it.
- If the information is not present in the context, say that it is not provided.
- Do not omit information that is available just because another part is missing.

Context:
{context}

Question:
{question}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text


@app.post("/ask")
def ask_question(question: str):

    if faiss_index is None:
        return {
            "answer": "Please upload a PDF first."
        }

    context = retrieve(question)

    answer = generate_answer(
        question,
        context
    )

    return {
        "answer": answer
    }
