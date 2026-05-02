from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from pathlib import Path


class Generator:
    def __init__(self, model_name="LiquidAI/LFM2-350M-GGUF", filename="LFM2-350M-Q4_K_M.gguf"):
        self.model_name = model_name
        self.filename = filename
        self.model_path = self.download_model()
        self.llm = Llama.from_pretrained(
            repo_id=self.model_name,
            filename=self.filename,
            n_ctx=8000,
            n_threads=4,
            verbose=False
        )

    def download_model(self):
        model_path = hf_hub_download(repo_id=self.model_name, filename=self.filename)
        return model_path

    def generate(self, prompt, max_tokens=256):
        output = self.llm(prompt, max_tokens=max_tokens)
        return output["choices"][0]["text"]

if __name__ == "__main__":
    query = "What are the regulations for cement production in India?"
    generator = Generator()
    output = generator.generate(query, max_tokens=256)
    print(output)