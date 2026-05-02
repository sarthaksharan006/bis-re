import json


def filter_short_chunks(chunks, word_threshold):
    # Keep chunks with short content.
    return [entry for entry in chunks if len(entry.get("content", "").split()) < word_threshold]


def filter_no_category(chunks):
    # Keep chunks that have no category yet.
    return [entry for entry in chunks if "category" not in entry or not entry.get("category")]


if __name__ == "__main__":
    with open("data/generated_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    word_threshold = 10
    short = filter_short_chunks(chunks, word_threshold)
    no_category = filter_no_category(chunks)

    # Save both filtered groups in one file.
    out = {
        "contents": short,
        "no_category": no_category
    }

    with open("data/filtered_chunks.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Filtered chunks (< {word_threshold} words): {len(short)}")
    print(f"Chunks without category: {len(no_category)}")
    print("Saved to data/filtered_chunks.json.\nManual editing needed. Rename edited file to chunks.json")