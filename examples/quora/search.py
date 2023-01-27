import argparse
import json
import string
from contextlib import suppress
from pathlib import Path

import ir_datasets
import numpy

import tinysearch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tinysearch_filename", type=Path)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    searcher = tinysearch.util.spinner("Loading searcher")(tinysearch.load)(args.tinysearch_filename)

    with suppress(KeyboardInterrupt, EOFError):
        while True:
            query = input("Query: ")
            for rank, (doc, score) in enumerate(searcher.search(query, topk=args.topk, return_scores=True), start=1):
                print(f"{rank:2d}. {doc['id']:6s} - {score:.4f} - {doc['text']}")
                print(f"    {' '.join(searcher.storage[doc['id'] + '__analyzed']['tokens'])}")
            print()


if __name__ == "__main__":
    main()
