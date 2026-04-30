import re
import json


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_id(block):
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
    text = block.replace("\n", " ")

    # remove SUMMARY OF
    text = re.sub(r"SUMMARY OF", "", text, flags=re.IGNORECASE)

    # remove ID
    text = re.sub(
        r"IS\s*\d+\s*(?:\(\s*Part\s*\d+\s*\))?\s*:\s*\d{4}",
        "",
        text,
        flags=re.IGNORECASE
    )

    # ---------------------------
    # TITLE = before first bullet
    # ---------------------------
    bullets = list(re.finditer(r"\b\d+\.\s", text))

    if bullets:
        title = text[:bullets[0].start()]
    else:
        title = text

    # ---------------------------
    # FIND SCOPE USING "Scope"
    # ---------------------------
    scope = ""

    scope_match = re.search(r"\bScope\b", text, re.IGNORECASE)

    if scope_match:
        start = scope_match.start()

        # try finding next top-level bullet
        next_bullet = re.search(r"\b\d+\.\s", text[start + 1:])

        if next_bullet:
            end = start + 1 + next_bullet.start()
            scope = text[start:end]
        else:
            # 🔥 FALLBACK: take next ~50 words
            words = text[start:].split()
            scope = " ".join(words[:50])

    else:
        # ---------------------------
        # fallback → use 1.
        # ---------------------------
        start_1 = None
        end_scope = None

        for i, b in enumerate(bullets):
            if b.group().startswith("1."):
                start_1 = b.start()

                if i + 1 < len(bullets):
                    end_scope = bullets[i + 1].start()
                break

        if start_1 is not None:
            if end_scope:
                scope = text[start_1:end_scope]
            else:
                # 🔥 fallback here too
                words = text[start_1:].split()
                scope = " ".join(words[:50])

    # ---------------------------
    # CLEAN TITLE
    # ---------------------------
    title = re.sub(r"\(.*?Revision.*?\)", "", title)
    title = re.sub(r"\s+", " ", title)

    return clean(title), clean(scope)


def build_chunks(text):
    blocks = text.split("SUMMARY OF")[1:]

    data = []

    for block in blocks:
        std_id = extract_id(block)
        if not std_id:
            continue

        title, scope = extract_title_and_scope(block)

        # fallback if scope missing
        if scope:
            content = f"{title} {scope}"
        else:
            content = block[:1000]

        # remove everything after next section (extra safety)
        content = re.sub(r"\b2\..*", "", content)

        content = clean(content)

        data.append({
            "id": std_id,
            "content": content
        })

    return data


if __name__ == "__main__":
    with open("data/raw_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = build_chunks(text)

    with open("data/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print("Chunks created:", len(chunks))