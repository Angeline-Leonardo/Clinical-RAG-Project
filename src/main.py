from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np 
import faiss 
app = FastAPI()

index = faiss.read_index("src/index.faiss")
model = SentenceTransformer("all-MiniLM-L6-V2")
docs = np.load("src/docs.npy", allow_pickle=True)
llm_name= "google/flan-t5-base" # can switch to bigger medical reasoning model on GPU
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request:QueryRequest):

    query_embedding = model.encode([request.query])
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    distances, indices = index.search(query_embedding, k=1)
    top_idx = int(indices[0][0])
    doc = docs[top_idx]

    context = doc
    prompt = f"""
    You are a medical assistant, use this context to answer the question.
    context: {context}
    question: {request.query}
    answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "query": request.query,
        "context": doc,
        "generated answer": answer
    } 
