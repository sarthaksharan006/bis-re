import gradio as gr
from src.retriever import retrieve
from src.generator import Generator

print("Loading generator model...")
generator = Generator()
print("Model loaded!")

def chat(message, history):
    retrieved_info = retrieve(message)

    prompt = f"Context information is below.\n---------------------\n{retrieved_info}\n---------------------\nGiven the context information, answer the following query: {message}\n\nAnswer:"
    
    prediction = generator.generate(prompt, max_tokens=512)
    return prediction


demo = gr.ChatInterface(
    fn=chat,
    examples=["I love cement", "What are the regulations for cement production?"],
    title="BIS Regulation Chatbot",
    description="A RAG chatbot that provides information about BIS regulations. Ask me anything about BIS regulations, and I'll do my best to help you!"
)

if __name__ == "__main__":
    demo.launch()
