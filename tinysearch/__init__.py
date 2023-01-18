import math
from importlib.metadata import version
from os import PathLike
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union, cast

from tinysearch import util
from tinysearch.storages import Storage
from tinysearch.tinysearch import TinySearch
from tinysearch.typing import DenseMatrix, Document, SparseMatrix

__version__ = version("tinysearch")


def bm25(
    documents: Iterable[Document],
    *,
    id_field: str = "id",
    text_field: str = "text",
    batch_size: int = 1000,
    approximate_search: bool = False,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    stopwords: Optional[Sequence[str]] = None,
) -> TinySearch[Document, SparseMatrix]:
    from tinysearch.indexers import AnnSparseIndexer, SparseIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import BM25Vectorizer
    from tinysearch.vocabulary import Vocabulary

    vocab = Vocabulary()
    storage = documents if isinstance(documents, Storage) else storage or MemoryStorage()
    analyzer = analyzer or (lambda text: text.split())
    for document in util.progressbar(documents, desc="Loading documents"):
        docid = document[id_field]
        docid_analyzed = f"{docid}__analyzed"
        if docid not in storage:
            storage[docid] = document
        if docid_analyzed not in storage:
            storage[docid_analyzed] = cast(Document, {"id": docid, "tokens": analyzer(document[text_field])})
        vocab.add_document(storage[docid_analyzed]["tokens"])

    def iter_analyzed_texts() -> Iterable[Tuple[str, Sequence[str]]]:
        assert storage is not None
        for docid, doc in storage.items():
            if docid.endswith("__analyzed"):
                docid = docid[:-10]
                yield docid, doc["tokens"]

    indexer: Union[SparseIndexer, AnnSparseIndexer]
    if approximate_search:
        indexer = AnnSparseIndexer("dotprod")
    else:
        indexer = SparseIndexer("dotprod")

    vectorizer = BM25Vectorizer(vocab)

    num_batches = math.ceil(len(storage) / batch_size)
    for batch in util.progressbar(
        util.batched(iter_analyzed_texts(), batch_size),
        total=num_batches,
        desc="Indexing documents",
    ):
        ids, docs = zip(*batch)
        vectors = vectorizer.vectorize_documents(docs)
        indexer.insert(ids, vectors)

    if isinstance(indexer, AnnSparseIndexer):
        print("Building ANN indexer...")
        indexer.build(print_progress=True)

    return TinySearch(
        storage=storage,
        indexer=indexer,
        vectorizer=vectorizer,
        analyzer=analyzer,
        stopwords=stopwords,
    )


def tfidf(
    documents: Iterable[Document],
    *,
    id_field: str = "id",
    text_field: str = "text",
    batch_size: int = 1000,
    similarity: str = "cosine",
    approximate_search: bool = False,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    stopwords: Optional[Sequence[str]] = None,
) -> TinySearch[Document, SparseMatrix]:
    from tinysearch.indexers import AnnSparseIndexer, SparseIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import TfidfVectorizer
    from tinysearch.vocabulary import Vocabulary

    vocab = Vocabulary()
    storage = documents if isinstance(documents, Storage) else storage or MemoryStorage()
    analyzer = analyzer or (lambda text: text.split())
    for document in util.progressbar(documents, desc="Loading documents"):
        docid = document[id_field]
        docid_analyzed = f"{docid}__analyzed"
        if docid not in storage:
            storage[docid] = document
        if docid_analyzed not in storage:
            storage[docid_analyzed] = cast(Document, {"id": docid, "tokens": analyzer(document[text_field])})
        vocab.add_document(storage[docid_analyzed]["tokens"])

    def iter_analyzed_texts() -> Iterable[Tuple[str, Sequence[str]]]:
        assert storage is not None
        for docid, doc in storage.items():
            if docid.endswith("__analyzed"):
                docid = docid[:-10]
                yield docid, doc["tokens"]

    indexer: Union[SparseIndexer, AnnSparseIndexer]
    if approximate_search:
        indexer = AnnSparseIndexer(space=similarity)
    else:
        indexer = SparseIndexer(space=similarity)

    vectorizer = TfidfVectorizer(vocab)

    num_batches = math.ceil(len(storage) / batch_size)
    for batch in util.progressbar(
        util.batched(iter_analyzed_texts(), batch_size),
        total=num_batches,
        desc="Indexing documents",
    ):
        ids, docs = zip(*batch)
        vectors = vectorizer.vectorize_documents(docs)
        indexer.insert(ids, vectors)

    if isinstance(indexer, AnnSparseIndexer):
        print("Building ANN indexer...")
        indexer.build(print_progress=True)

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
    *,
    id_field: str = "id",
    text_field: str = "text",
    probabilities: Optional[Mapping[str, float]] = None,
    similarity: str = "cosine",
    smoothing: float = 1e-3,
    batch_size: int = 1000,
    approximate_search: bool = False,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    stopwords: Optional[Sequence[str]] = None,
) -> TinySearch[Document, DenseMatrix]:

    from tinysearch.indexers import AnnDenseIndexer, DenseIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import SifVectorizer
    from tinysearch.vocabulary import Vocabulary

    if isinstance(embeddings, (str, PathLike)):
        embeddings = util.spinner(desc=f"Loading embeddings from {embeddings}")(util.load_pretrained_embeddings)(
            embeddings
        )

    vocab = Vocabulary() if probabilities is None else None
    storage = documents if isinstance(documents, Storage) else storage or MemoryStorage()
    analyzer = analyzer or (lambda text: text.split())
    for document in util.progressbar(documents, desc="Loading documents"):
        docid = document[id_field]
        docid_analyzed = f"{docid}__analyzed"
        if docid not in storage:
            storage[docid] = document
        if docid_analyzed not in storage:
            storage[docid_analyzed] = cast(Document, {"id": docid, "tokens": analyzer(document[text_field])})
        if vocab is not None:
            vocab.add_document(storage[docid_analyzed]["tokens"])

    def iter_analyzed_texts() -> Iterable[Tuple[str, Sequence[str]]]:
        assert storage is not None
        for docid, doc in storage.items():
            if docid.endswith("__analyzed"):
                docid = docid[:-10]
                yield docid, doc["tokens"]

    if probabilities is None:
        assert vocab is not None
        probabilities = {token: vocab.get_token_probability(token) for token in vocab.token_to_index}

    indexer: Union[DenseIndexer, AnnDenseIndexer]
    if approximate_search:
        indexer = AnnDenseIndexer(space=similarity)
    else:
        indexer = DenseIndexer(space=similarity)

    vectorizer = SifVectorizer(probabilities, embeddings, smoothing=smoothing)

    num_batches = math.ceil(len(storage) / batch_size)
    for batch in util.progressbar(
        util.batched(iter_analyzed_texts(), batch_size),
        total=num_batches,
        desc="Indexing documents",
    ):
        ids, docs = zip(*batch)
        vectors = vectorizer.vectorize_documents(docs)
        indexer.insert(ids, vectors)

    if isinstance(indexer, AnnDenseIndexer):
        print("Building ANN indexer...")
        indexer.build(print_progress=True)

    return TinySearch(
        storage=storage,
        indexer=indexer,
        vectorizer=vectorizer,
        analyzer=analyzer,
        stopwords=stopwords,
    )


def swem(
    documents: Iterable[Document],
    embeddings: Union[str, PathLike, Mapping[str, DenseMatrix]],
    *,
    id_field: str = "id",
    text_field: str = "text",
    similarity: str = "cosine",
    window_size: int = 3,
    smoothing: float = 1e-3,
    batch_size: int = 1000,
    approximate_search: bool = False,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    stopwords: Optional[Sequence[str]] = None,
) -> TinySearch[Document, DenseMatrix]:

    from tinysearch.indexers import AnnDenseIndexer, DenseIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import SwemVectorizer

    if isinstance(embeddings, (str, PathLike)):
        embeddings = util.spinner(desc=f"Loading embeddings from {embeddings}")(util.load_pretrained_embeddings)(
            embeddings
        )

    storage = documents if isinstance(documents, Storage) else storage or MemoryStorage()
    analyzer = analyzer or (lambda text: text.split())
    for document in util.progressbar(documents, desc="Loading documents"):
        docid = document[id_field]
        docid_analyzed = f"{docid}__analyzed"
        if docid not in storage:
            storage[docid] = document
        if docid_analyzed not in storage:
            storage[docid_analyzed] = cast(Document, {"id": docid, "tokens": analyzer(document[text_field])})

    def iter_analyzed_texts() -> Iterable[Tuple[str, Sequence[str]]]:
        assert storage is not None
        for docid, doc in storage.items():
            if docid.endswith("__analyzed"):
                docid = docid[:-10]
                yield docid, doc["tokens"]

    indexer: Union[DenseIndexer, AnnDenseIndexer]
    if approximate_search:
        indexer = AnnDenseIndexer(space=similarity)
    else:
        indexer = DenseIndexer(space=similarity)

    vectorizer = SwemVectorizer(embeddings, window_size=window_size, smoothing=smoothing)

    num_batches = math.ceil(len(storage) / batch_size)
    for batch in util.progressbar(
        util.batched(iter_analyzed_texts(), batch_size),
        total=num_batches,
        desc="Indexing documents",
    ):
        ids, docs = zip(*batch)
        vectors = vectorizer.vectorize_documents(docs)
        indexer.insert(ids, vectors)

    if isinstance(indexer, AnnDenseIndexer):
        print("Building ANN indexer...")
        indexer.build(print_progress=True)

    return TinySearch(
        storage=storage,
        indexer=indexer,
        vectorizer=vectorizer,
        analyzer=analyzer,
        stopwords=stopwords,
    )
