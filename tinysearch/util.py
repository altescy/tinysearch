from typing import Callable, Iterable, Iterator, List, TypeVar

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
