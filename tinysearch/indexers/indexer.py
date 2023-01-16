import abc
import pickle
from os import PathLike
from typing import List, Optional, Sequence, Type, TypeVar, Union

from scipy import sparse

Self = TypeVar("Self", bound="Indexer")


class Indexer(abc.ABC):
    @abc.abstractmethod
    def insert(self, ids: Sequence[str], data: sparse.csr_matrix, update: bool = False) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def search(self, queries: sparse.csr_matrix, topk: Optional[int]) -> List[List[str]]:
        raise NotImplementedError

    def save(self, filename: Union[str, PathLike]) -> None:
        with open(filename, "wb") as pklfile:
            pickle.dump(self, pklfile)

    @classmethod
    def load(cls: Type[Self], filename: Union[str, PathLike]) -> Self:
        with open(filename, "rb") as pklfile:
            indexer = pickle.load(pklfile)
            assert isinstance(indexer, cls)
        return indexer
