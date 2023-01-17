from typing import Generic, List, Optional, Sequence, TypeVar

from tinysearch.indexers import Indexer
from tinysearch.storages import Storage
from tinysearch.typing import Analyzer, Document, Matrix
from tinysearch.vectorizers import Vectorizer
from tinysearch.vocabulary import Vocabulary

Self = TypeVar("Self", bound="TinySearch")


class TinySearch(Generic[Document, Matrix]):
    def __init__(
        self,
        storage: Storage[Document],
        vocab: Vocabulary,
        indexer: Indexer[Matrix],
        vectorizer: Vectorizer[Matrix],
        analyzer: Analyzer,
        stopwords: Optional[Sequence[str]] = None,
    ) -> None:
        self.storage: Storage[Document] = storage
        self.vocab: Vocabulary = vocab
        self.indexer: Indexer[Matrix] = indexer
        self.vectorizer: Vectorizer[Matrix] = vectorizer
        self.analyzer = analyzer
        self.stopwords = set(stopwords or ())

    def search(self, query: str, topk: Optional[int] = 10) -> List[Document]:
        tokens = list(self.analyzer(query))
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        query_vector = self.vectorizer.vectorize_queries([tokens])
        ids = self.indexer.search(query_vector, topk=topk)[0]
        return [self.storage[id_] for id_ in ids]
