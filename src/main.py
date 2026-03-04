from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np 
import faiss 
app = FastAPI()

index = faiss.read_index("src/index.faiss")
model = SentenceTransformer("all-MiniLM-L6-V2")
docs = np.load("src/docs.npy", allow_pickle=True)
llm_name= "google/gemma-7b-it" # can switch to bigger medical reasoning model on GPU
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_name)
class QueryRequest(BaseModel):
    query: str

# ask : simple identitfication of patients 
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
    outputs = llm_model.generate(**inputs, max_new_tokens=10)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "query": request.query,
        "context": doc,
        "generated answer": answer
    } 

# classify : classify diff patients into stable, emergency, critical 
@app.get("/classify")
def classify_patient():
    docs = np.load("src/docs.npy", allow_pickle=True)
    prompt=f"""You are a clinical triage assistant. Classify these patients into
    Stable, Emergency or Critical. {docs}.
    Answer only in this format: Patient ID: Status. 
    Example: Patient P003: Stable 
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=150, do_sample=False, temperature=0.0)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer



