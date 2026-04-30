import fitz

def extract_pdf_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        blocks = page.get_text("blocks")

        # sort blocks top-to-bottom
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        for b in blocks:
            all_text.append(b[4])

    return "\n".join(all_text)


if __name__ == "__main__":
    text = extract_pdf_text_blocks("data/dataset.pdf")

    with open("data/raw_text.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("Saved improved text to raw_text_v2.txt")