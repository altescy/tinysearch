from typing import Generic, Iterable, List, Optional, Sequence, Type, TypeVar

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

    @classmethod
    def from_documents(
        cls: Type[Self],
        documents: Iterable[Document],
        batch_size: int = 1000,
        storage: Optional[Storage[Document]] = None,
        indexer: Optional[Indexer[Matrix]] = None,
        vectorizer: Optional[Vectorizer[Matrix]] = None,
        analyzer: Optional[Analyzer] = None,
        stopwords: Optional[Sequence[str]] = None,
    ) -> Self:
        from tinysearch import util
        from tinysearch.indexers import ScipyIndexer
        from tinysearch.storages import MemoryStorage
        from tinysearch.vectorizers import BM25Vectorizer

        storage = storage or MemoryStorage()
        for document in documents:
            storage[document["id"]] = document

        analyzer = analyzer or (lambda text: text.split())
        analyzed_documents = {doc["id"]: analyzer(doc["text"]) for doc in documents}

        vocab = Vocabulary.from_documents(analyzed_documents.values())
        indexer = indexer or ScipyIndexer(len(vocab))
        vectorizer = vectorizer or BM25Vectorizer(vocab)

        for batch in util.batched(analyzed_documents.items(), batch_size):
            ids, docs = zip(*batch)
            vectors = vectorizer.vectorize_documents(docs)
            indexer.insert(ids, vectors)

        return cls(
            storage=storage,
            vocab=vocab,
            indexer=indexer,
            vectorizer=vectorizer,
            analyzer=analyzer,
            stopwords=stopwords,
        )
