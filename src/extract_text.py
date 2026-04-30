import fitz  # pymupdf
from tqdm import tqdm

pdf_path = "data/dataset.pdf"
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    print(f"Total pages: {len(doc)}")

    for page in tqdm(doc, desc="Reading PDF"):
        text = page.get_text()
        all_text.append(text)

    return "\n".join(all_text)


if __name__ == "__main__":
    pdf_path = "data/dataset.pdf"

    text = extract_pdf_text(pdf_path)

    # Save raw text (VERY useful for debugging)
    with open("data/raw_text.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("Text extraction complete. Saved to data/raw_text.txt")