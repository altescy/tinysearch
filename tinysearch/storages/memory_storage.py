from typing import Dict, Iterator, Tuple, TypeVar

from tinysearch.storages.storage import Storage

T = TypeVar("T")


class MemoryStorage(Storage[T]):
    def __init__(self) -> None:
        self._storage: Dict[str, T] = {}

    def __getitem__(self, key: str) -> T:
        return self._storage[key]

    def __setitem__(self, key: str, value: T) -> None:
        self._storage[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._storage

    def __iter__(self) -> Iterator[str]:
        return iter(self._storage.keys())

    def __len__(self) -> int:
        return len(self._storage)

    def keys(self) -> Iterator[str]:
        return iter(self._storage.keys())

    def values(self) -> Iterator[T]:
        return iter(self._storage.values())

    def items(self) -> Iterator[Tuple[str, T]]:
        return iter(self._storage.items())
