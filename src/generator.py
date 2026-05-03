from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from src.constants import SYSTEM_PROMPT, MODEL_NAME, FILENAME
from .logger import get_logger

LOGGER = get_logger(__name__)


def load_model(model_name=MODEL_NAME, filename=FILENAME):
    try:
        LOGGER.info("Trying to find model locally...")
        hf_hub_download(repo_id=model_name, filename=filename, local_files_only=True)
        LOGGER.info("Loaded model from local cache.")
    except Exception as e:
        LOGGER.info("Model not found locally. Downloading...")
        try:
            hf_hub_download(repo_id=model_name, filename=filename)
            LOGGER.info("Model downloaded and cached.")
        except Exception as download_error:
            LOGGER.error("Error occurred while downloading model: %s", str(download_error))
            raise

    return Llama.from_pretrained(
        repo_id=model_name,
        filename=filename,
        n_ctx=8000,
        n_threads=4,
        verbose=False,
    )


LLM = load_model()


def generate(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=256):
    # Manually format the string into LFM's exact ChatML structure
    try:
        LOGGER.info("Generating response for prompt: %s", prompt)
        formatted_prompt = (
            f"<|startoftext|><|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        output = LLM(formatted_prompt, max_tokens=max_tokens)
        return output["choices"][0]["text"]
    except Exception as e:
        LOGGER.error("Error occurred during generation: %s", str(e))
        raise RuntimeError("Generation failed") from e
