import json
import time
import argparse
from tqdm import tqdm
from src.retriever import retrieve
from src.logger import get_logger


LOGGER = get_logger(__name__)

# This script evaluates retrieval quality in bulk.
# It reads a JSON list of queries, runs retrieve() for each, and writes results
# (including latency and expected vs. retrieved standards) to an output file.

LOGGER.info("Running inference...")


def main(input_path, output_path):
    # Load input queries .json file.
    with open(input_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = []

    # tqdm provides a progress bar with elapsed time and ETA.
    for item in tqdm(
        queries,
        total=len(queries),
        desc="Retrieving",
        unit=" query",
        dynamic_ncols=True,
    ):
        # Measure per-query latency for evaluation/reporting.
        start = time.time()

        query = item["query"]
        # Control output size: change k=5 to desired number of results.
        retrieved = retrieve(query, k=5)

        latency = time.time() - start

        # Persist the full record so offline evaluation can compare expected vs actual.
        results.append({
            "id": item["id"],
            "query": item["query"],
            "expected_standards": item.get("expected_standards", []),
            "retrieved_standards": retrieved,
            "latency_seconds": round(latency, 4)
        })

    # Write a JSON array of results for later scoring/analysis.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    LOGGER.info("Inference completed. Processed %d queries.", len(queries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    main(args.input, args.output)