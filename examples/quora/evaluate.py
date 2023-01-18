import argparse
import json
import string
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Set, Tuple, Union, cast

import ir_datasets
import numpy
from metrics import NDCG, FMeasure, MultiMetrics

import tinysearch


class SimpleAnalyzer:
    def __call__(self, text: str) -> List[str]:
        text = text.lower()
        for punct in string.punctuation:
            text = text.replace(punct, " ")
        return text.split()


class DatasetReader:
    def __init__(self, name: str) -> None:
        self.dataset = ir_datasets.load(name)

    def load_documents(self) -> Iterator[Dict[str, Any]]:
        for doc in self.dataset.docs_iter():
            yield {"id": doc.doc_id, "text": doc.text}

    def load_query(self) -> Iterator[Dict[str, Any]]:
        for query in self.dataset.queries_iter():
            yield {"id": query.query_id, "text": query.text}

    def load_relations(self) -> Dict[Tuple[str, str], int]:
        relations: Dict[Tuple[str, str], int] = {}
        for qrel in self.dataset.qrels_iter():
            relations[(qrel.query_id, qrel.doc_id)] = qrel.relevance
        return relations

    def load_golds(self) -> Dict[str, List[Tuple[str, int]]]:
        gold: Dict[str, List[Tuple[str, int]]] = {}
        for qrel in self.dataset.qrels_iter():
            if qrel.query_id not in gold:
                gold[qrel.query_id] = []
            gold[qrel.query_id].append((qrel.doc_id, qrel.relevance))
        return gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["dev", "test"], default="dev")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    dataset_reader = DatasetReader(f"beir/quora/{args.subset}")
    searcher = tinysearch.bm25(
        dataset_reader.load_documents(),
        analyzer=SimpleAnalyzer(),
        approximate_search=True,
    )

    metrics = MultiMetrics(NDCG(args.topk), FMeasure())
    relations = dataset_reader.load_relations()
    golds = dataset_reader.load_golds()
    for i, query in enumerate(dataset_reader.load_query()):
        search_results = searcher.search(query["text"], topk=args.topk)
        gold = golds[query["id"]]
        pred = [(doc["id"], relations.get((query["id"], doc["id"]), 0)) for doc in search_results]
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
