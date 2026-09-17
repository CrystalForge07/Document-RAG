
import faiss, os
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

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"),base_url="https://api.groq.com/openai/v1")

# Retrieval (and augmentation)

def retrieve(question):

    question_embedding = embedding_model.encode([question],normalize_embeddings=True)

    top_k = 5

    distances, indices = faiss_index.search(question_embedding,top_k)

    context = "\n\n".join(
    f"[Page {chunks[chunk_index]['page']}]\n"
    f"{chunks[chunk_index]['text']}"
    for chunk_index in indices[0]
    )

    return context

def load_and_chunk_pdf(pdf_path):

    pdf = pymupdf.open(pdf_path)

    chunks = []

    chunk_size = 200
    overlap = 40

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text()
        words = text.split()

        for i in range(0, len(words), chunk_size - overlap):

            chunk_words = words[i:i + chunk_size]

            if not chunk_words:
                continue

            chunk = {
                "text": " ".join(chunk_words),
                "page": page_number
            }

            chunks.append(chunk)

    pdf.close()

    return chunks

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    global chunks, faiss_index

    with open("uploaded.pdf", "wb") as buffer:
        buffer.write(file.file.read())

    chunks = load_and_chunk_pdf("uploaded.pdf")

    embeddings = embedding_model.encode([chunk["text"] for chunk in chunks],normalize_embeddings=True)

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)

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

    At the end, list the page numbers used in the context under "Sources".

    Context:
    {context}

    Question:
    {question}
    """

    response = client.responses.create(model="openai/gpt-oss-120b",input=prompt)

    return response.output_text

@app.post("/ask")
def ask_question(question: str):

    if faiss_index is None:
        return {"answer": "Please upload a PDF first." }

    context = retrieve(question)

    answer = generate_answer(question,context)

    return {"answer": answer}
