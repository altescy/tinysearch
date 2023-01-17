from collections import defaultdict
from os import PathLike
from typing import Dict, Iterable, Iterator, List, TypeVar, Union

import numpy

T = TypeVar("T")


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_pretrained_embedding(filename: Union[str, PathLike]) -> Dict[str, numpy.ndarray]:
    embeddings: Dict[str, numpy.ndarray] = {}
    with open(filename, "r") as textfile:
        for line in textfile:
            parts = line.split()
            word = parts[0]
            embedding = numpy.array([float(x) for x in parts[1:]])
            embeddings[word] = embedding

    return defaultdict(lambda: numpy.zeros(len(embedding)), embeddings)
