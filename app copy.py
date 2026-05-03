import gradio as gr
from src.logger import get_logger
from src.retriever import retrieve_for_llm
from src.generator import generate

# app.py is the UI entry point for the RAG chatbot.
# It wires the Gradio interface to the retrieval + generation pipeline:
# - retrieve_for_llm() (src/retriever.py) pulls relevant chunks from FAISS.
# - generate() (src/generator.py) runs the local GGUF LLM with a system prompt.
# The output is then rendered in the chat UI and the retrieved context panel.

LOGGER = get_logger(__name__)

LOGGER.info("Loading generator model...")
LOGGER.info("Model loaded!")

def chat(message, history):
    # UI entry point: map a user message -> context retrieval -> LLM response.
    # This is the single function Gradio calls for both button click and Enter.
    # Step 1: retrieve a list of chunk dicts (standard/category/content).
    retrieved_info = retrieve_for_llm(message)
    # Step 2: format the retrieved chunks into a human-readable context block.
    # This same block is shown in the UI AND fed into the LLM prompt below.
    context_text = "\n\n".join(
        [
            "\n".join(
                [
                    f"Standard: {chunk.get('standard', '')}",
                    f"Category: {chunk.get('category', '')}",
                    f"Content: {chunk.get('content', '')}",
                ]
            )
            for chunk in retrieved_info
        ]
    )

    # Step 3: build the user prompt; generate() will prepend SYSTEM_PROMPT.
    prompt = f"Context:\n{context_text}\n\nQuestion: {message}\n\nAnswer:"
    # Step 4: call the local LLM and get a plain-text response.
    prediction = generate(prompt, max_tokens=512)
    # Step 5: extend chat history in the format Gradio's Chatbot expects.
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": prediction},
    ]
    return history, context_text


with gr.Blocks() as demo:
    # Gradio UI layout that wires inputs to the chat() handler above.
    gr.Markdown("# BIS Regulation Chatbot")
    gr.Markdown(
        "A RAG chatbot that provides information about BIS regulations. "
        "Ask me anything about BIS regulations, and I'll do my best to help you!"
    )

    # Primary chat display for the conversation transcript.
    chatbot = gr.Chatbot()
    with gr.Row():
        # Message input field and a Send button live side-by-side.
        message = gr.Textbox(
            label="Message",
            placeholder="Type your question here...",
            scale=3,
        )
        submit = gr.Button("Send", scale=1)

    # Side panel shows the retrieved chunks used for LLM grounding and debugging.
    retrieved_box = gr.Textbox(
        label="Retrieved Chunks",
        lines=12,
        interactive=False,
    )

    # Example prompts that auto-fill the input box for quick testing.
    examples = gr.Examples(
        examples=["I love cement", "What are the regulations for cement production?"],
        inputs=message,
    )

    # Hook both button click and Enter key to the same inference path.
    # The function returns (chat history, retrieved context) to update two widgets.
    submit.click(chat, inputs=[message, chatbot], outputs=[chatbot, retrieved_box])
    message.submit(chat, inputs=[message, chatbot], outputs=[chatbot, retrieved_box])

if __name__ == "__main__":
    # Launch a local web server for the chatbot UI.
    demo.launch()
