import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-V2")
with open("data/clinical_notes.json", "r") as f:
    clinical_data = json.load(f)

notes = [item["note"] for item in clinical_data]


embeddings = model.encode(notes) 
embeddings = np.asarray(embeddings, dtype=np.float32) 


dimension = embeddings.shape[1]
index= faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "index.faiss")
np.save("docs.npy", np.array(clinical_data, dtype=object))
print("Success")