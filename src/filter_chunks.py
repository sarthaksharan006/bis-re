import json


def filter_short_chunks(chunks, word_threshold):
    filtered_chunks = []
    for entry in chunks:
        content = entry.get("content", "")
        word_count = len(content.split())
        if word_count < word_threshold:
            filtered_chunks.append(entry)
    return filtered_chunks


if __name__ == "__main__":
    with open("data/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    word_threshold = 15
    filtered_chunks = filter_short_chunks(chunks, word_threshold)

    with open("data/filtered_chunks.json", "w", encoding="utf-8") as f:
        json.dump(filtered_chunks, f, indent=2, ensure_ascii=False)

    print(f"Filtered chunks (< {word_threshold} words): {len(filtered_chunks)}")
    print("Saved to data/filtered_chunks.json")