import pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
import faiss
import pickle

# Converting pdf into plain text

def load_and_chunk_pdf(pdf_path):
    pdf = pymupdf.open(pdf_path)
    chunks = []
    
    for page in pdf:
        text = page.get_text()
        words = text.split()

        chunk_size = 200
        overlap = 40

        for i in range(0, len(words), chunk_size-overlap):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)

    pdf.close()

    return chunks

# Embeddings
def create_embeddings(chunks, model):
    embeddings = model.encode(chunks)
    return embeddings

# Asking a question and also encoding it

def retrieve(question, model, faiss_index, chunks):

    question_embedding = model.encode([question])

    top_k = 5

    distances, indices = faiss_index.search(
        question_embedding,
        top_k
    )

    print("\nRetrieved chunks:")

    for i, chunk_index in enumerate(indices[0]):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunks[chunk_index])

    context = "\n\n".join(
        chunks[chunk_index] for chunk_index in indices[0]
    )

    return context

def generate_answer(question, context, client):

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

# Main program

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

pdf_path = input("Enter PDF path : ")
chunks = load_and_chunk_pdf(pdf_path)

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

embeddings = create_embeddings(
    chunks,
    embedding_model
)

np.save("embeddings.npy",embeddings)

dimension = embeddings.shape[1]

faiss_index = faiss.IndexFlatL2(dimension)

faiss_index.add(embeddings)

faiss.write_index(faiss_index, "faiss_index.bin")

while True:

    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    context = retrieve(
        question,
        embedding_model,
        faiss_index,
        chunks
    )

    answer = generate_answer(
        question,
        context,
        client
    )

    print("\nAnswer:")
    print(answer)
