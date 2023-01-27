import abc
from typing import Generic, Iterator, Tuple, TypeVar

T = TypeVar("T")


class Storage(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def __getitem__(self, key: str) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def __setitem__(self, key: str, value: T) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def __contains__(self, key: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def keys(self) -> Iterator[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def values(self) -> Iterator[T]:
        raise NotImplementedError

    def items(self) -> Iterator[Tuple[str, T]]:
        return zip(self.keys(), self.values())

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
