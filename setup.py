print("Loading packages...")
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"

def download_model():
    print(f"Downloading model: {MODEL_NAME}")
    SentenceTransformer(MODEL_NAME)
    print("Model download complete.")


if __name__ == "__main__":
    download_model()
