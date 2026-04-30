import re
import json


def clean(text):
    # Just squashes weird spacing so the output is easier to read.
    return re.sub(r"\s+", " ", text).strip()


def extract_id(block):
    # The standard ID usually shows up near the top of the block, so we search the whole thing.
    block = block.replace("\n", " ")

    match = re.search(
        r"IS\s*(\d+)\s*(?:\(\s*Part\s*(\d+)\s*\))?\s*:\s*(\d{4})",
        block,
        re.IGNORECASE
    )

    if not match:
        return None

    base = match.group(1)
    part = match.group(2)
    year = match.group(3)

    if part:
        return f"IS {base} (Part {part}): {year}"
    else:
        return f"IS {base}: {year}"


def extract_title_and_scope(block):
    # Flatten the text so the bullet parsing works even when the source has line breaks everywhere.
    text = block.replace("\n", " ")

    # We look for numbered bullets like 1., 2., 3. and use those as anchors.
    bullets = list(re.finditer(r"\b(\d+)\.\s", text))

    title = ""
    scope = ""

    if bullets:
        # Whatever comes before the first bullet is treated as the title.
        first_bullet = bullets[0]

        title = text[:first_bullet.start()].strip()

    else:
        # If there are no bullets at all, just keep the text as the title and move on.
        return clean(text), ""

    start_1 = None
    start_2 = None

    for b in bullets:
        num = b.group(1)

        if num == "1" and start_1 is None:
            start_1 = b.start()

        elif num == "2" and start_1 is not None:
            start_2 = b.start()
            break

    # Scope starts at 1. and usually ends right before 2. if a second section exists.
    if start_1 is not None:
        if start_2:
            scope = text[start_1:start_2]
        else:
            scope = text[start_1:]

    return clean(title), clean(scope)


def build_chunks(text):
    # The raw file is split into sections using the repeated "SUMMARY OF" heading.
    blocks = text.split("SUMMARY OF")[1:]

    data = []

    for block in blocks:
        # Skip any block that does not look like a standard entry.
        std_id = extract_id(block)
        if not std_id:
            continue

        title, scope = extract_title_and_scope(block)

        # Store the useful bits together so the JSON stays simple to use later.
        content = f"{title} {scope}"

        data.append({
            "id": std_id,
            "content": content
        })

    return data


if __name__ == "__main__":
    # Read the big raw text dump, turn it into chunks, then write the JSON output.
    with open("data/raw_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = build_chunks(text)

    with open("data/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print("Chunks created:", len(chunks))