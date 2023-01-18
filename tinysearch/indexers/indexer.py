import abc
from typing import Generic, List, Optional, Sequence, Tuple

from tinysearch.typing import Matrix


class Indexer(abc.ABC, Generic[Matrix]):
    @abc.abstractmethod
    def insert(self, ids: Sequence[str], data: Matrix, update: bool = False) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def search(self, queries: Matrix, topk: Optional[int]) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError
