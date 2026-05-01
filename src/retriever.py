import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
print("Running retriever...")

def load_model(model_name="BAAI/bge-base-en-v1.5"):
    try:
        print("Trying to load model locally...")
        model = SentenceTransformer(model_name, local_files_only=True)
        print("Loaded model from local cache")
        return model

    except Exception as e:
        print("Model not found locally. Downloading...")
        model = SentenceTransformer(model_name)
        print("Model downloaded and cached")
        return model


# load everything once
model = load_model()
index = faiss.read_index("data/index.faiss")

with open("data/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)


def retrieve(query, k=5):
    query = query.lower()

    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    D, I = index.search(q_emb, k)

    results = []
    for idx in I[0]:
        results.append(data[idx]["standard"])

    # remove duplicates
    results = list(dict.fromkeys(results))

    return results[:k] 
    
print("Retrieving..")