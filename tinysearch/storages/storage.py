import abc
from typing import Generic, Iterator, TypeVar

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
    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError
