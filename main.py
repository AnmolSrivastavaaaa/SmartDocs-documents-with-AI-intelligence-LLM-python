import os
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# FastAPI + templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB setup
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="documents")

# Uploads folder
os.makedirs("uploads", exist_ok=True)

# Local GPT2 setup
GPT_MODEL = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL)
gpt_model = AutoModelForCausalLM.from_pretrained(GPT_MODEL)

# Keep track of uploaded files
uploaded_files = []

# Home page
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "uploaded_files": uploaded_files}
    )

# Upload document
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    # Add file to uploaded list if not already there
    if file.filename not in uploaded_files:
        uploaded_files.append(file.filename)

    # Read text
    text = open(path, "r", encoding="utf-8").read()
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Save chunks to ChromaDB
    for idx, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            metadatas=[{"filename": file.filename, "chunk_index": idx}],
            embeddings=[embedding],
            ids=[f"{file.filename}_{idx}"]
        )
    return RedirectResponse("/", status_code=303)

# Ask question
@app.post("/ask")
def ask_question(request: Request, question: str = Form(...)):
    # 1️⃣ Find nearest chunk
    q_emb = model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=1)
    context = results["documents"][0][0] if results["documents"][0] else "No context available."

    # 2️⃣ Generate answer locally with GPT2
    prompt = f"Answer the question using this context:\nContext: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = gpt_model.generate(**inputs, max_new_tokens=100)
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    answer = gen_text.split("Answer:")[-1].strip()
    if len(answer) > 500:
        answer = answer[:500] + "..."

    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "answer": answer, 
            "uploaded_files": uploaded_files,
            "source": f"{results['metadatas'][0][0]['filename']}" if results["metadatas"][0] else None
        }
    )
