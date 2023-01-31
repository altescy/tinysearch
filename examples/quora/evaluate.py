import argparse
import json
import string
import time
from pathlib import Path

import ir_datasets
import numpy
from src.dataloader import DataLoader
from src.metrics import NDCG, FMeasure, MultiMetrics

import tinysearch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tinysearch_filename", type=Path)
    parser.add_argument("--subset", choices=["dev", "test"], default="dev")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    dataset_reader = DataLoader(f"beir/quora/{args.subset}")
    searcher = tinysearch.util.spinner("Loading models")(tinysearch.load)(args.tinysearch_filename)

    metrics = MultiMetrics(NDCG(args.topk), FMeasure(args.topk))
    relations = dataset_reader.load_relations()
    golds = dataset_reader.load_golds()

    num_done = 0
    elapsed_time = 0.0
    for batch in tinysearch.util.batched(dataset_reader.load_query(), args.batch_size):
        queries = [query["text"] for query in batch]
        start_time = time.time()
        search_results = searcher.search(queries, topk=args.topk)
        elapsed_time += time.time() - start_time
        for query, result in zip(batch, search_results):
            gold = golds[query["id"]]
            pred = [(doc["id"], relations[(query["id"], doc["id"])]) for doc in result]
            metrics(gold, pred)
            num_done += 1
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.get_metrics().items())
            print(f"\r{100 * num_done/len(golds):6.2f}% speed={num_done/elapsed_time:.4f}qs/s {metrics_str}", end="")

    print()

    evaluation_result = {f"{k}@{args.topk}": v for k, v in metrics.get_metrics().items()}
    max_namelen = max(len(k) for k in evaluation_result.keys())
    print("Evaluation result:")
    print("  Average queries/s:", len(golds) / elapsed_time)
    print("  Metrics:")
    for name, value in evaluation_result.items():
        print(f"    {name:<{max_namelen}s}: {value:.4f}")


if __name__ == "__main__":
    main()
