# EMBEDDING CONSTANTS
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# GENERATION CONSTANTS
MODEL_NAME = "LiquidAI/LFM2-350M-GGUF"

FILENAME = "LFM2-350M-Q4_K_M.gguf"

SYSTEM_PROMPT = """You are an automated extraction engine specializing in Bureau of Indian Standards (BIS) regulations. 
Your only task is to process 5 text chunks from BIS IS documents.

For each chunk, you must extract:
1. The official code of the IS regulation.
2. The name of the standard and a concise, 1-2 sentence summary of what the technical standard or regulation covers.

Extract only factual, technical details. Be direct and do not add conversational filler."""