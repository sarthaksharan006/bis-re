import re
import fitz
from .logger import get_logger


LOGGER = get_logger(__name__)


def extract_pdf_text_blocks(pdf_path):
    # Extract raw text from a PDF by reading each page's text blocks and
    # re-ordering them top-to-bottom, left-to-right for more natural reading.
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        blocks = page.get_text("blocks")

        # Each block is a tuple; b[1] is the top y-coordinate, b[0] is the left x-coordinate.
        # Sorting by (y, x) approximates human reading order.
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        for b in blocks:
            # b[4] is the text payload of the block.
            all_text.append(b[4])

    return "\n".join(all_text)


def cleanup_text(text):
    # Strategy: trim preamble content so downstream parsing starts near SECTION 1,
    # then remove known boilerplate tokens that pollute chunking.
    section_heading = re.compile(r"(?im)^\s*SECTION\s+1\b.*$")
    matches = list(section_heading.finditer(text))

    if len(matches) >= 2:
        # PDF repeats SECTION 1; prefer the second occurrence to skip headers.
        cleaned = text[matches[1].start():]
    elif len(matches) == 1:
        # If only one match exists, start from the first SECTION 1.
        cleaned = text[matches[0].start():]
    else:
        # If no section marker is found, keep the full text as-is.
        cleaned = text

    # Remove a known catalog/reference string that appears in the PDF footer/header.
    cleaned = cleaned.replace("SP  21 : 2005", "")
    return cleaned


if __name__ == "__main__":
    LOGGER.info("Extracting text from dataset.pdf...")
    # Pipeline: extract raw block text -> clean -> write to data/raw.txt for chunking.
    text = extract_pdf_text_blocks("data/dataset.pdf")
    text = cleanup_text(text)

    with open("data/raw.txt", "w", encoding="utf-8") as f:
        f.write(text)

    LOGGER.info("Saved text to data/raw.txt")