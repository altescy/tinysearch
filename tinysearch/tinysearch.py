import pickle
from os import PathLike
from typing import Generic, List, Literal, Optional, Tuple, Type, TypeVar, Union, overload

from tinysearch.indexers import Indexer
from tinysearch.storages import Storage
from tinysearch.typing import Analyzer, Document, Matrix
from tinysearch.vectorizers import Vectorizer

Self = TypeVar("Self", bound="TinySearch")


class TinySearch(Generic[Document, Matrix]):
    def __init__(
        self,
        storage: Storage[Document],
        indexer: Indexer[Matrix],
        vectorizer: Vectorizer[Matrix],
        analyzer: Analyzer,
    ) -> None:
        self.storage: Storage[Document] = storage
        self.indexer: Indexer[Matrix] = indexer
        self.vectorizer: Vectorizer[Matrix] = vectorizer
        self.analyzer = analyzer

    @overload
    def search(
        self,
        query: str,
        *,
        topk: Optional[int] = ...,
    ) -> List[Document]:
        ...

    @overload
    def search(
        self,
        query: str,
        *,
        return_scores: Literal[False],
        topk: Optional[int] = ...,
    ) -> List[Document]:
        ...

    @overload
    def search(
        self,
        query: str,
        *,
        return_scores: Literal[True],
        topk: Optional[int] = ...,
    ) -> List[Tuple[Document, float]]:
        ...

    @overload
    def search(
        self,
        query: List[str],
        *,
        topk: Optional[int] = ...,
    ) -> List[List[Document]]:
        ...

    @overload
    def search(
        self,
        query: List[str],
        *,
        return_scores: Literal[False],
        topk: Optional[int] = ...,
    ) -> List[List[Document]]:
        ...

    @overload
    def search(
        self,
        query: List[str],
        *,
        return_scores: Literal[True],
        topk: Optional[int] = ...,
    ) -> List[List[Tuple[Document, float]]]:
        ...

    def search(
        self,
        query: Union[str, List[str]],
        *,
        return_scores: bool = False,
        topk: Optional[int] = 10,
    ) -> Union[List[Document], List[Tuple[Document, float]], List[List[Document]], List[List[Tuple[Document, float]]]]:
        return_as_batch = True
        if isinstance(query, str):
            query = [query]
            return_as_batch = False

        batched_tokens = [self.analyzer(q) for q in query]
        query_vector = self.vectorizer.vectorize_queries(batched_tokens)
        results = self.indexer.search(query_vector, topk=topk)

        output: Union[List[List[Document]], List[List[Tuple[Document, float]]]]
        if return_scores:
            output = [[(self.storage[id_], score) for id_, score in result] for result in results]
        else:
            output = [[self.storage[id_] for id_, _ in result] for result in results]
        if return_as_batch:
            return output
        return output[0]

    def save(self, filename: Union[str, PathLike]) -> None:
        with open(filename, "wb") as pklfile:
            pickle.dump(self, pklfile)

    @classmethod
    def load(cls: Type[Self], filename: Union[str, PathLike]) -> Self:
        with open(filename, "rb") as pklfile:
            searcher = pickle.load(pklfile)
            if not isinstance(searcher, cls):
                raise TypeError(f"Expected type {cls.__name__}, got {type(searcher).__name__}")
        return searcher
