print("Loading packages...")
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

from src.constants import EMBEDDING_MODEL_NAME, MODEL_NAME, FILENAME


def download_embedder():
    print(f"Downloading embedding model: {EMBEDDING_MODEL_NAME}")
    SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model download complete.")


def download_gguf():
    print(f"Downloading GGUF model: {MODEL_NAME} / {FILENAME}")
    hf_hub_download(repo_id=MODEL_NAME, filename=FILENAME)
    print("GGUF model download complete.")


if __name__ == "__main__":
    download_embedder()
    answer = input("Download optional LLM model now? [y/N]: ").strip().lower()
    if answer in {"y", "yes"}:
        download_gguf()
    else:
        print("Skipping GGUF model download.")
