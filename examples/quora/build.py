import argparse
import json
import string
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Set, Tuple, Union, cast

import ir_datasets
import minato
import numpy
from src.dataloader import DataLoader
from src.metrics import NDCG, FMeasure, MultiMetrics

import tinysearch
from tinysearch.analyzers import NltkAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-filename", type=Path, default=None)
    parser.add_argument("--indexer", choices=["bm25", "sif", "swem"], default="bm25")
    args = parser.parse_args()

    dataset_reader = DataLoader(f"beir/quora")

    output_filename: Path
    if args.indexer == "bm25":
        searcher = tinysearch.bm25(
            dataset_reader.load_documents(),
            analyzer=NltkAnalyzer(stemmer="porter"),
        )
        output_filename = args.output_filename or (Path("outputs") / "bm25.pkl")
    elif args.indexer == "sif":
        searcher = tinysearch.sif(
            dataset_reader.load_documents(),
            embeddings=minato.cached_path("https://nlp.stanford.edu/data/glove.6B.zip!glove.6B.300d.txt"),
            approximate_search=True,
            analyzer=NltkAnalyzer(),
            indexer_config={"method": "napp", "space_params": {"numPivot": 1000}},
        )
        output_filename = args.output_filename or (Path("outputs") / "sif.pkl")
    elif args.indexer == "swem":
        searcher = tinysearch.swem(
            dataset_reader.load_documents(),
            embeddings=minato.cached_path("https://nlp.stanford.edu/data/glove.6B.zip!glove.6B.300d.txt"),
            approximate_search=True,
            analyzer=NltkAnalyzer(),
            indexer_config={"method": "napp", "space_params": {"numPivot": 1000}},
        )
        output_filename = args.output_filename or (Path("outputs") / "swem.pkl")
    else:
        raise ValueError(f"Unknown indexer: {args.indexer}")

    print("Saving index to", output_filename)
    searcher.save(output_filename)


if __name__ == "__main__":
    main()
