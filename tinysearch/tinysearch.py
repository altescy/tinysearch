from typing import Generic, List, Literal, Optional, Sequence, Tuple, TypeVar, Union, overload

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
        stopwords: Optional[Sequence[str]] = None,
    ) -> None:
        self.storage: Storage[Document] = storage
        self.indexer: Indexer[Matrix] = indexer
        self.vectorizer: Vectorizer[Matrix] = vectorizer
        self.analyzer = analyzer
        self.stopwords = set(stopwords or ())

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

    def search(
        self,
        query: str,
        *,
        return_scores: bool = False,
        topk: Optional[int] = 10,
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        tokens = list(self.analyzer(query))
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        query_vector = self.vectorizer.vectorize_queries([tokens])
        results = self.indexer.search(query_vector, topk=topk)[0]
        if return_scores:
            return [(self.storage[id_], score) for id_, score in results]
        return [self.storage[id_] for id_, _ in results]
