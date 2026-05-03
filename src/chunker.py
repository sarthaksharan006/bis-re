import re
import json
from .logger import get_logger


LOGGER = get_logger(__name__)

# This module turns raw BIS text into structured "chunks" for retrieval.
# High-level flow:
# 1) Scan the raw text to map standard IDs (IS numbers) -> section/category labels.
# 2) Split the text into SUMMARY OF blocks and extract a title + scope per block.
# 3) Write a list of chunk dicts with "standard", "content", and "category".


def clean(text):
    # Normalize whitespace so downstream parsing and embeddings are stable.
    # This collapses newlines/tabs into single spaces and trims ends.
    return re.sub(r"\s+", " ", text).strip()


def is_all_caps_line(line):
    # Treat fully uppercase lines with at least one letter as headings.
    # This ignores lines that are just numbers or punctuation.
    stripped = line.strip()
    if not stripped:
        return False
    letters = re.findall(r"[A-Za-z]", stripped)
    return bool(letters) and stripped == stripped.upper()


def build_category_map(text):
    # Walk the raw text once and remember the current section/subsection.
    # The goal is to map IS numbers like "IS 456" -> "SECTION 3 CEMENT" (example).
    lines = text.splitlines()
    section_heading = re.compile(r"^\s*SECTION\s+\d+\s*$", re.IGNORECASE)

    # Mapping string ID number -> string category label.
    id_to_category = {}
    # Remembering current section and subsection titles as we scan lines.
    section_name = None
    subsection_name = None
    # Flags to control state while we scan for section headings.
    looking_for_section_name = False
    in_section = False

    # Manual index loop so we can skip ahead when we consume multi-line headings.
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            # Skip empty lines; they do not carry semantic meaning here.
            i += 1
            continue

        if section_heading.match(line):
            # Found "SECTION <number>" line; the real title typically follows.
            in_section = True
            looking_for_section_name = True
            section_name = None
            subsection_name = None
            i += 1
            continue

        if re.search(r"\bSUMMARY OF\b", line, re.IGNORECASE):
            # SUMMARY OF marks the start of the actual standards list.
            # After this, section headers are no longer meaningful for mapping.
            in_section = False
            looking_for_section_name = False
            subsection_name = None
            i += 1
            continue

        if in_section and looking_for_section_name:
            # Section name may continue onto the next ALL-CAPS line(s).
            if is_all_caps_line(line):
                # Accumulate a multi-line ALL-CAPS section title.
                parts = [line]
                # Keep joining nearby ALL-CAPS lines until the name ends.
                j = i + 1
                while j < len(lines):
                    nxt = lines[j].strip()
                    if not nxt:
                        break
                    if nxt.upper() == "CONTENTS":
                        break
                    if is_all_caps_line(nxt):
                        parts.append(nxt)
                        j += 1
                        continue
                    break

                # Final section title is the joined ALL-CAPS lines.
                section_name = " ".join(p for p in parts if p)
                looking_for_section_name = False
                # advance index to last consumed line
                i = j
                continue
            i += 1
            continue

        if in_section and section_name and is_all_caps_line(line):
            if line.upper() != "CONTENTS" and not re.match(r"^IS\b", line, re.IGNORECASE):
            # Treat uppercase lines as subsection names within a section.
                subsection_name = line
                i += 1
                continue

        if in_section and section_name:
            # Look for a standard identifier line (starts with "IS").
            id_number = None
            if re.match(r"^IS\b", line, re.IGNORECASE):
                same_line = re.search(r"^IS\s*(\d+)", line, re.IGNORECASE)
                if same_line:
                    # The ID number appears on the same line as the "IS" token.
                    id_number = same_line.group(1)
                elif line.upper() == "IS":
                    # Sometimes "IS" is on its own line; the number is next.
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines):
                        next_line = lines[j].strip()
                        m = re.match(r"^(\d+)", next_line)
                        if m:
                            id_number = m.group(1)
                            i = j

            if id_number:
                # Map the standard number to the current section/subsection.
                if subsection_name:
                    category = f"{section_name} {subsection_name}"
                else:
                    category = section_name

                if id_number not in id_to_category:
                    # Only keep the first mapping if the ID repeats.
                    id_to_category[id_number] = category

        i += 1

    return id_to_category


def extract_id(block):
    # Extract a canonical standard identifier from a SUMMARY OF block.
    # Example matches:
    #   "IS 456: 2000"
    #   "IS 456 (Part 2): 2015"
    block = block.replace("\n", " ")

    match = re.search(
        r"IS\s*(\d+)\s*(?:\(\s*Part\s*(\d+)\s*\))?\s*:\s*(\d{4})",
        block,
        re.IGNORECASE
    )

    if not match:
        # Without a valid ID, this block is not a standard entry.
        return None

    base = match.group(1)
    part = match.group(2)
    year = match.group(3)

    if part:
        return f"IS {base} (Part {part}): {year}"
    else:
        return f"IS {base}: {year}"


def extract_title_and_scope(block):
    # Pull a usable "title" and "scope" from a SUMMARY OF block.
    # This is heuristic parsing; it assumes BIS formatting conventions.
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
    # TITLE = before first bullet OR "Scope"
    # ---------------------------

    # Identify bullet markers like "1." and the word "Scope".
    bullets = list(re.finditer(r"\b\d+\.\s", text))
    scope_match = re.search(r"\bScope\b", text, re.IGNORECASE)

    title_end_candidates = []

    # first bullet position
    if bullets:
        title_end_candidates.append(bullets[0].start())

    # scope position
    if scope_match:
        title_end_candidates.append(scope_match.start())

    if title_end_candidates:
        title_end = min(title_end_candidates)
        title = text[:title_end]
    else:
        title_end = len(text)
        title = text

    scope = ""

    # ---------------------------
    # FIND SCOPE USING "Scope"
    # ---------------------------

    if scope_match:
        start = scope_match.start()

        next_bullet = re.search(r"\b\d+\.\s", text[start + 1:])

        if next_bullet:
            # Scope ends at the next bullet section.
            end = start + 1 + next_bullet.start()
            scope = text[start:end]
        else:
            # Otherwise take a short window (first 50 words) after "Scope".
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
                # Use bullet 1 to bullet 2 as the scope text.
                scope = text[start_1:end_scope]
            else:
                # If only bullet 1 exists, take a small word window after it.
                words = text[start_1:].split()
                scope = " ".join(words[:50])

        else:
            # FINAL FALLBACK: use the first 50 words after the title.
            words = text[title_end:].split()
            scope = " ".join(words[:50])

    # ---------------------------
    # CLEAN TITLE
    # ---------------------------
    # Remove parenthetical "Revision" notes and normalize whitespace.
    title = re.sub(r"\(.*?Revision.*?\)", "", title)
    title = re.sub(r"\s+", " ", title)

    return clean(title), clean(scope)

def build_chunks(text):
    # Split each SUMMARY OF block into one chunk entry.
    LOGGER.info("Building chunks...")
    blocks = text.split("SUMMARY OF")[1:]

    data = []

    for block in blocks:
        # Each block should represent a single standard summary.
        std_id = extract_id(block)
        if not std_id:
            # Skip malformed blocks without a detectable ID.
            continue

        title, scope = extract_title_and_scope(block)

        # fallback if scope missing
        if scope:
            # Normal case: combine title and scope into one searchable string.
            content = f"{title} {scope}"
        else:
            # If scope parsing fails, keep a raw slice of the block.
            content = block[:1000]

        # remove everything after next section (extra safety)
        content = re.sub(r"\b2\..*", "", content)

        # Final cleanup ensures consistent whitespace in embeddings.
        content = clean(content)

        data.append({
            "standard": std_id,
            "content": content
        })

    LOGGER.info("Built %d chunks.", len(data))
    return data


if __name__ == "__main__":
    # Script mode: build chunks and attach categories in one pass.
    with open("data/raw.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = build_chunks(text)
    id_to_category = build_category_map(text)

    for entry in chunks:
        # Attach the parsed category back onto the matching chunk.
        match = re.search(r"\bIS\s*(\d+)", entry.get("standard", ""))
        if match:
            # Use the numeric ID (e.g., 456) to look up its section/subsection.
            category = id_to_category.get(match.group(1))
            if category:
                entry["category"] = category

    with open("data/generated_chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print("Chunks created:", len(chunks))