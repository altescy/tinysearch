import argparse
import json
import string
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Set, Tuple, Union, cast

import ir_datasets
import numpy
from src.analyzer import SimpleAnalyzer
from src.dataloader import DataLoader
from src.metrics import NDCG, FMeasure, MultiMetrics

import tinysearch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_filename", type=Path)
    parser.add_argument("--subset", choices=["dev", "test"], default="dev")
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    args = parser.parse_args()

    dataset_reader = DataLoader(f"beir/quora/{args.subset}")
    searcher = tinysearch.bm25(
        dataset_reader.load_documents(),
        analyzer=SimpleAnalyzer(),
    )

    searcher.save(args.output_filename)


if __name__ == "__main__":
    main()