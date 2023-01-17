from typing import Generic, List, Optional, Sequence, Tuple, TypeVar, Union

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

    def search(
        self,
        query: str,
        topk: Optional[int] = 10,
        return_scores: bool = False,
        return_documents: bool = True,
    ) -> Union[List[str], List[Tuple[str, float]], List[Document], List[Tuple[Document, float]]]:
        tokens = list(self.analyzer(query))
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        query_vector = self.vectorizer.vectorize_queries([tokens])
        results = self.indexer.search(query_vector, topk=topk)[0]
        if return_scores and return_documents:
            return [(self.storage[id_], score) for id_, score in results]
        if return_scores and not return_documents:
            return results
        if not return_scores and return_documents:
            return [self.storage[id_] for id_, _ in results]
        return [id_ for id_, _ in results]
