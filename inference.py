import json
import time
import argparse
from src.retriever import retrieve


def main(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = []

    for item in queries:
        start = time.time()

        query = item["query"]
        retrieved = retrieve(query, k=5)

        latency = time.time() - start

        results.append({
            "id": item["id"],
            "query": item["query"],
            "expected_standards": item.get("expected_standards", []),
            "retrieved_standards": retrieved,
            "latency_seconds": round(latency, 4)
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    main(args.input, args.output)