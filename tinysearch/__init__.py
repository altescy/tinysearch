from importlib.metadata import version
from typing import Callable, Iterable, Optional, Sequence

from tinysearch.storages import Storage
from tinysearch.tinysearch import TinySearch
from tinysearch.typing import Document, SparseMatrix

__version__ = version("tinysearch")


def bm25(
    documents: Iterable[Document],
    batch_size: int = 1000,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    stopwords: Optional[Sequence[str]] = None,
) -> TinySearch[Document, SparseMatrix]:
    from tinysearch import util
    from tinysearch.indexers import ScipyIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import BM25Vectorizer
    from tinysearch.vocabulary import Vocabulary

    storage = storage or MemoryStorage()
    for document in documents:
        storage[document["id"]] = document

    analyzer = analyzer or (lambda text: text.split())
    analyzed_documents = {doc["id"]: analyzer(doc["text"]) for doc in documents}

    vocab = Vocabulary.from_documents(analyzed_documents.values())
    indexer = ScipyIndexer(len(vocab))
    vectorizer = BM25Vectorizer(vocab)

    for batch in util.batched(analyzed_documents.items(), batch_size):
        ids, docs = zip(*batch)
        vectors = vectorizer.vectorize_documents(docs)
        indexer.insert(ids, vectors)

    return TinySearch(
        storage=storage,
        indexer=indexer,
        vectorizer=vectorizer,
        analyzer=analyzer,
        stopwords=stopwords,
    )
