import gradio as gr
from src.logger import get_logger
from src.retriever import retrieve_for_llm
from src.generator import generate

LOGGER = get_logger(__name__)

LOGGER.info("Loading generator model...")
LOGGER.info("Model loaded!")

def chat(message, history):
    retrieved_info = retrieve_for_llm(message)
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

    prompt = f"Context:\n{context_text}\n\nQuestion: {message}\n\nAnswer:"
    prediction = generate(prompt, max_tokens=512)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": prediction},
    ]
    return history, context_text


with gr.Blocks() as demo:
    gr.Markdown("# BIS Regulation Chatbot")
    gr.Markdown(
        "A RAG chatbot that provides information about BIS regulations. "
        "Ask me anything about BIS regulations, and I'll do my best to help you!"
    )

    chatbot = gr.Chatbot()
    with gr.Row():
        message = gr.Textbox(
            label="Message",
            placeholder="Type your question here...",
            scale=3,
        )
        submit = gr.Button("Send", scale=1)

    retrieved_box = gr.Textbox(
        label="Retrieved Chunks",
        lines=12,
        interactive=False,
    )

    examples = gr.Examples(
        examples=["I love cement", "What are the regulations for cement production?"],
        inputs=message,
    )

    submit.click(chat, inputs=[message, chatbot], outputs=[chatbot, retrieved_box])
    message.submit(chat, inputs=[message, chatbot], outputs=[chatbot, retrieved_box])

if __name__ == "__main__":
    demo.launch()
