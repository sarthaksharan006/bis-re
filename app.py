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


# ---------------------------------------------------------------------------
# Gruvbox palette (dark hard variant)
# ---------------------------------------------------------------------------
# bg:    #1d2021   bg1: #3c3836   bg2: #504945   bg3: #665c54
# fg:    #ebdbb2   fg1: #d5c4a1   fg2: #bdae93
# red:   #fb4934   green: #b8bb26  yellow: #fabd2f
# blue:  #83a598   purple: #d3869b  aqua:  #8ec07c  orange: #fe8019

GRUVBOX_CSS = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Zilla+Slab:wght@400;600;700&display=swap');

/* ── CSS Variables ── */
:root {
    --gb-bg:      #1d2021;
    --gb-bg0:     #282828;
    --gb-bg1:     #3c3836;
    --gb-bg2:     #504945;
    --gb-bg3:     #665c54;
    --gb-bg4:     #7c6f64;
    --gb-fg:      #ebdbb2;
    --gb-fg1:     #d5c4a1;
    --gb-fg2:     #bdae93;
    --gb-red:     #fb4934;
    --gb-green:   #b8bb26;
    --gb-yellow:  #fabd2f;
    --gb-blue:    #83a598;
    --gb-purple:  #d3869b;
    --gb-aqua:    #8ec07c;
    --gb-orange:  #fe8019;
    --radius:     6px;
    --font-mono:  'JetBrains Mono', 'Fira Code', monospace;
    --font-serif: 'Zilla Slab', Georgia, serif;
}

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background-color: var(--gb-bg) !important;
    color: var(--gb-fg) !important;
    font-family: var(--font-mono) !important;
}

/* ── Outer wrapper ── */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 24px 20px 40px !important;
}

/* ── Header block ── */
.app-header {
    border-left: 4px solid var(--gb-orange);
    padding: 6px 0 6px 16px;
    margin-bottom: 8px;
}

/* ── Gradio Markdown ── */
.gr-markdown h1, .prose h1,
div[class*="markdown"] h1 {
    font-family: var(--font-serif) !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--gb-yellow) !important;
    letter-spacing: 0.5px;
    margin: 0 0 4px 0 !important;
    line-height: 1.2;
}

div[class*="markdown"] p,
.gr-markdown p, .prose p {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    color: var(--gb-fg2) !important;
    margin: 0 !important;
    line-height: 1.6;
}

/* ── Accent rule under header ── */
.header-divider {
    height: 1px;
    background: linear-gradient(90deg, var(--gb-orange) 0%, var(--gb-yellow) 40%, transparent 100%);
    margin: 14px 0 20px 0;
    border: none;
}

/* ── Chatbot ── */
.gr-chatbot, [class*="chatbot"] {
    background-color: var(--gb-bg0) !important;
    border: 1px solid var(--gb-bg2) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
}

/* User bubble */
[class*="chatbot"] [class*="user"] > div,
[class*="message"][data-testid="user"] {
    background-color: var(--gb-bg2) !important;
    color: var(--gb-fg) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--gb-bg3) !important;
    font-family: var(--font-mono) !important;
}

/* Assistant bubble */
[class*="chatbot"] [class*="bot"] > div,
[class*="message"][data-testid="bot"] {
    background-color: var(--gb-bg1) !important;
    color: var(--gb-aqua) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--gb-bg2) !important;
    font-family: var(--font-mono) !important;
}

/* ── Textboxes & Inputs ── */
textarea, input[type="text"], .gr-textbox textarea {
    background-color: var(--gb-bg0) !important;
    color: var(--gb-fg) !important;
    border: 1px solid var(--gb-bg2) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s ease;
    caret-color: var(--gb-orange);
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--gb-orange) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(254,128,25,0.18) !important;
}

textarea::placeholder, input::placeholder {
    color: var(--gb-bg4) !important;
}

/* Retrieved chunks panel — monospace terminal feel */
#retrieved-panel textarea {
    background-color: var(--gb-bg) !important;
    color: var(--gb-green) !important;
    border-color: var(--gb-bg2) !important;
    font-size: 0.78rem !important;
    line-height: 1.6;
}

/* ── Labels ── */
label > span, .gr-label, [class*="label"] span {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    color: var(--gb-bg4) !important;
}

/* ── Send Button ── */
#send-btn {
    background-color: var(--gb-orange) !important;
    color: var(--gb-bg) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    padding: 0 20px !important;
    height: 44px !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease, transform 0.1s ease;
}

#send-btn:hover {
    background-color: var(--gb-yellow) !important;
    transform: translateY(-1px);
}

#send-btn:active {
    transform: translateY(0);
}

/* ── Examples ── */
.gr-examples, [class*="examples"] {
    margin-top: 8px !important;
}

.gr-examples button, [class*="examples"] button {
    background-color: var(--gb-bg1) !important;
    color: var(--gb-fg2) !important;
    border: 1px solid var(--gb-bg2) !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    padding: 4px 10px !important;
    transition: border-color 0.15s, color 0.15s;
}

.gr-examples button:hover, [class*="examples"] button:hover {
    border-color: var(--gb-orange) !important;
    color: var(--gb-orange) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--gb-bg0); }
::-webkit-scrollbar-thumb { background: var(--gb-bg2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gb-bg3); }

/* ── Panel section labels (custom) ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--gb-bg4);
    margin-bottom: 6px;
    padding-left: 2px;
}

/* ── Status bar at bottom ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 18px;
    padding: 6px 12px;
    background: var(--gb-bg0);
    border: 1px solid var(--gb-bg1);
    border-radius: var(--radius);
    font-size: 0.7rem;
    color: var(--gb-bg4);
}

.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--gb-green);
    box-shadow: 0 0 5px var(--gb-green);
}

/* ── Responsive tweak ── */
@media (max-width: 640px) {
    .gradio-container { padding: 12px 10px 24px !important; }
    div[class*="markdown"] h1 { font-size: 1.4rem !important; }
}
"""


def chat(message, history):
    """UI entry point: map a user message -> context retrieval -> LLM response."""
    # Step 1: retrieve a list of chunk dicts (standard/category/content).
    retrieved_info = retrieve_for_llm(message)

    # Step 2: format the retrieved chunks into a human-readable context block.
    context_text = "\n\n".join(
        [
            "\n".join(
                [
                    f"Standard : {chunk.get('standard', '')}",
                    f"Category : {chunk.get('category', '')}",
                    f"Content  : {chunk.get('content', '')}",
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


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="BIS Regulation Chatbot") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    with gr.Column(elem_classes=["app-header"]):
        gr.Markdown("# BIS Regulation Chatbot")
        gr.Markdown(
            "Retrieval-augmented answers over BIS standards. "
            "Ask a question — the system will surface relevant regulations and ground the response."
        )
    gr.HTML('<hr class="header-divider">')

    # ── Main chat area ───────────────────────────────────────────────────────
    chatbot = gr.Chatbot(
        label="Conversation",
        height=300,
        show_label=True,
    )

    # ── Input row ────────────────────────────────────────────────────────────
    with gr.Row():
        message = gr.Textbox(
            label="",
            placeholder="Ask about BIS regulations…",
            scale=5,
            lines=1,
            max_lines=4,
            show_label=False,
        )
        submit = gr.Button("Send ↩", scale=1, elem_id="send-btn")

    # ── Examples ─────────────────────────────────────────────────────────────
    gr.Examples(
        examples=[
            "What are the regulations for cement production?",
            "What BIS standards apply to electrical wiring?",
            "I love cement",
        ],
        inputs=message,
        label="Quick prompts",
    )

    # ── Retrieved chunks panel ───────────────────────────────────────────────
    retrieved_box = gr.Textbox(
        label="Retrieved Chunks",
        lines=10,
        interactive=False,
        elem_id="retrieved-panel",
        placeholder="Retrieved context will appear here after your first query…",
    )

    # ── Status bar ───────────────────────────────────────────────────────────
    gr.HTML(
        '<div class="status-bar">'
        '<div class="status-dot"></div>'
        '<span>Model loaded &nbsp;·&nbsp; FAISS index ready &nbsp;·&nbsp; Local inference</span>'
        '</div>'
    )

    # ── Event wiring ─────────────────────────────────────────────────────────
    submit.click(chat, inputs=[message, chatbot], outputs=[chatbot, retrieved_box])
    message.submit(chat, inputs=[message, chatbot], outputs=[chatbot, retrieved_box])


if __name__ == "__main__":
    demo.launch(css=GRUVBOX_CSS)