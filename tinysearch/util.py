import functools
import itertools
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from os import PathLike
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sized, TextIO, TypeVar, Union

import numpy
from scipy import sparse

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


def progressbar(
    iterable: Iterable[T],
    total: Optional[int] = None,
    desc: Optional[str] = None,
    barlength: int = 20,
) -> Iterator[T]:
    if isinstance(iterable, Sized):
        total = total or len(iterable)

    def get_line(i: int) -> str:
        if total is not None:
            percentage = i / total
            bar = f"{'=' * int(percentage * barlength):{barlength}}"
            total_length = len(str(total))
            line = f"[{bar}] {i:>{total_length}}/{total}"
        else:
            line = f"{i}"
        if desc:
            line = f"{desc}: {line}"
        return line

    for i, item in enumerate(iterable):
        print("\r" + get_line(i), end="")
        yield item

    print("\r" + get_line(i + 1))


def spinner(
    desc: Optional[str] = None,
    *,
    chars: Iterable[str] = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
    delay: float = 0.1,
    file: TextIO = sys.stdout,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    if not isinstance(chars, itertools.cycle):
        chars = itertools.cycle(chars)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            def show_spinner(event: threading.Event) -> None:
                assert isinstance(chars, itertools.cycle)
                start = time.time()
                while not event.is_set():
                    elapsed = time.time() - start
                    message = f"\r{next(chars)} {msg} ... {elapsed:.2f}s"
                    print(message, end="", file=file, flush=True)
                    time.sleep(delay)

            event = threading.Event()
            msg = f"executing {func.__name__}" if desc is None else desc
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(show_spinner, event)
                try:
                    result = func(*args, **kwargs)
                finally:
                    event.set()
                    while not future.done():
                        pass
                    print(file=file, flush=True)

            return result

        return wrapper

    return decorator


def load_pretrained_embeddings(filename: Union[str, PathLike]) -> Dict[str, numpy.ndarray]:
    embeddings: Dict[str, numpy.ndarray] = {}
    with open(filename, "r") as textfile:
        for line in textfile:
            parts = line.split()
            word = parts[0]
            embedding = numpy.array([float(x) for x in parts[1:]])
            embeddings[word] = embedding

    return defaultdict(lambda: numpy.zeros(len(embedding)), embeddings)


def distance_to_similarity(space: str, distance: float) -> float:
    if space == "cosine":
        return 1.0 - distance
    if space in ("dotprod", "l1", "l2", "linf"):
        return -distance
    raise ValueError(f"Unknown space: {space}")


def csr_row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    matrix = matrix.copy()
    norms = numpy.array(sparse.linalg.norm(matrix, axis=1)).flatten()
    scale = numpy.reciprocal(norms, where=norms != 0)
    matrix.data *= numpy.repeat(scale, numpy.diff(matrix.indptr))
    matrix.eliminate_zeros()
    return matrix
