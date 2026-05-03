import re
import fitz
from .logger import get_logger


LOGGER = get_logger(__name__)


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


def cleanup_text(text):
    section_heading = re.compile(r"(?im)^\s*SECTION\s+1\b.*$")
    matches = list(section_heading.finditer(text))

    if len(matches) >= 2:
        cleaned = text[matches[1].start():]
    elif len(matches) == 1:
        cleaned = text[matches[0].start():]
    else:
        cleaned = text

    cleaned = cleaned.replace("SP  21 : 2005", "")
    return cleaned


if __name__ == "__main__":
    LOGGER.info("Extracting text from dataset.pdf...")
    text = extract_pdf_text_blocks("data/dataset.pdf")
    text = cleanup_text(text)

    with open("data/raw.txt", "w", encoding="utf-8") as f:
        f.write(text)

    LOGGER.info("Saved text to data/raw.txt")