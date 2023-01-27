import argparse
import json
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    dataset_reader = DataLoader(f"beir/quora/{args.subset}")
    searcher = tinysearch.util.spinner("Loading models")(tinysearch.load)(args.tinysearch_filename)

    metrics = MultiMetrics(NDCG(args.topk), FMeasure(args.topk))
    relations = dataset_reader.load_relations()
    golds = dataset_reader.load_golds()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(searcher.search, query["text"], topk=args.topk): query
            for query in dataset_reader.load_query()
        }
        for i, future in enumerate(as_completed(futures)):
            query = futures[future]
            search_results = future.result()
            gold = golds[query["id"]]
            pred = [(doc["id"], relations[(query["id"], doc["id"])]) for doc in search_results]
            metrics(gold, pred)
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.get_metrics().items())
            print(f"\r{100 * i/len(golds):6.2f}% {metrics_str}", end="")

    print()

    evaluation_result = {f"{k}@{args.topk}": v for k, v in metrics.get_metrics().items()}
    max_namelen = max(len(k) for k in evaluation_result.keys())
    print("Metrics:")
    for name, value in evaluation_result.items():
        print(f"  {name:<{max_namelen}s}: {value:.4f}")


if __name__ == "__main__":
    main()
