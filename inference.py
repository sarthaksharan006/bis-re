import json
import time
import argparse
from tqdm import tqdm
from src.retriever import retrieve
from src.logger import get_logger


LOGGER = get_logger(__name__)

LOGGER.info("Running inference...")


def main(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    results = []

    for item in tqdm(
        queries,
        total=len(queries),
        desc="Retrieving",
        unit=" query",
        dynamic_ncols=True,
    ):
        start = time.time()

        query = item["query"]
        # Control output size: change k=5 to desired number of results
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

    LOGGER.info("Inference completed. Processed %d queries.", len(queries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    main(args.input, args.output)