from importlib.metadata import version
from os import PathLike
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

from tinysearch import util
from tinysearch.storages import Storage
from tinysearch.tinysearch import TinySearch
from tinysearch.typing import DenseMatrix, Document, SparseMatrix

__version__ = version("tinysearch")


def bm25(
    documents: Iterable[Document],
    batch_size: int = 1000,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    stopwords: Optional[Sequence[str]] = None,
) -> TinySearch[Document, SparseMatrix]:
    from tinysearch.indexers import SparseIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import BM25Vectorizer
    from tinysearch.vocabulary import Vocabulary

    storage = storage or MemoryStorage()
    for document in documents:
        storage[document["id"]] = document

    analyzer = analyzer or (lambda text: text.split())
    analyzed_documents = {doc["id"]: analyzer(doc["text"]) for doc in documents}

    vocab = Vocabulary.from_documents(analyzed_documents.values())
    indexer = SparseIndexer(len(vocab))
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


def sif(
    documents: Iterable[Document],
    embeddings: Union[str, PathLike, Mapping[str, DenseMatrix]],
    smoothing: float = 1e-3,
    batch_size: int = 1000,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    stopwords: Optional[Sequence[str]] = None,
) -> TinySearch[Document, DenseMatrix]:

    from tinysearch.indexers import DenseIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import SifVectorizer
    from tinysearch.vocabulary import Vocabulary

    if isinstance(embeddings, (str, PathLike)):
        embeddings = util.load_pretrained_embedding(embeddings)

    storage = storage or MemoryStorage()
    for document in documents:
        storage[document["id"]] = document

    analyzer = analyzer or (lambda text: text.split())
    analyzed_documents = {doc["id"]: analyzer(doc["text"]) for doc in documents}

    vocab = Vocabulary.from_documents(analyzed_documents.values())
    indexer = DenseIndexer(len(vocab))
    vectorizer = SifVectorizer(vocab, embeddings, smoothing=smoothing)

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
