import math
from importlib.metadata import version
from os import PathLike
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union, cast

from tinysearch import util
from tinysearch.storages import Storage
from tinysearch.tinysearch import TinySearch
from tinysearch.typing import DenseMatrix, Document, SparseMatrix

__version__ = version("tinysearch")


def save(searcher: TinySearch, path: PathLike) -> None:
    searcher.save(path)


def load(path: PathLike) -> TinySearch:
    return TinySearch.load(path)


def bm25(
    documents: Iterable[Document],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    id_field: str = "id",
    text_field: str = "text",
    batch_size: int = 1000,
    approximate_search: bool = False,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    indexer_config: Optional[Mapping[str, Any]] = None,
    postprocessing_config: Optional[Mapping[str, Any]] = None,
) -> TinySearch[Document, SparseMatrix]:
    from tinysearch.analyzers import WhitespaceTokenizer
    from tinysearch.indexers import AnnSparseIndexer, InvertedIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import BM25Vectorizer
    from tinysearch.vocabulary import Vocabulary

    if isinstance(documents, Storage):
        storage = documents
    elif storage is None:
        storage = MemoryStorage()

    vocab = Vocabulary()
    analyzer = analyzer or WhitespaceTokenizer()
    num_documents = 0
    for document in util.progressbar(documents, desc="Loading documents"):
        num_documents += 1
        docid = document[id_field]
        docid_analyzed = f"{docid}__analyzed"
        if docid not in storage:
            storage[docid] = document
        if docid_analyzed not in storage:
            storage[docid_analyzed] = cast(Document, {"id": docid, "tokens": analyzer(document[text_field])})
        vocab.add_document(storage[docid_analyzed]["tokens"])

    storage.flush()

    def iter_analyzed_texts() -> Iterable[Tuple[str, Sequence[str]]]:
        assert storage is not None
        for docid, doc in storage.items():
            if docid.endswith("__analyzed"):
                docid = docid[:-10]
                yield docid, doc["tokens"]

    indexer: Union[InvertedIndexer, AnnSparseIndexer]
    if approximate_search:
        indexer = AnnSparseIndexer(space="dotprod", **(indexer_config or {}))
    else:
        indexer = InvertedIndexer(**(indexer_config or {}))

    vectorizer = BM25Vectorizer(vocab)

    num_batches = math.ceil(num_documents / batch_size)
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
        indexer.build(print_progress=True, **(postprocessing_config or {}))

    return TinySearch(
        storage=storage,
        indexer=indexer,
        vectorizer=vectorizer,
        analyzer=analyzer,
    )


def tfidf(
    documents: Iterable[Document],
    *,
    id_field: str = "id",
    text_field: str = "text",
    batch_size: int = 1000,
    approximate_search: bool = False,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    indexer_config: Optional[Mapping[str, Any]] = None,
    postprocessing_config: Optional[Mapping[str, Any]] = None,
) -> TinySearch[Document, SparseMatrix]:
    from tinysearch.analyzers import WhitespaceTokenizer
    from tinysearch.indexers import AnnSparseIndexer, InvertedIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import TfidfVectorizer
    from tinysearch.vocabulary import Vocabulary

    if isinstance(documents, Storage):
        storage = documents
    elif storage is None:
        storage = MemoryStorage()

    vocab = Vocabulary()
    analyzer = analyzer or WhitespaceTokenizer()
    num_documents = 0
    for document in util.progressbar(documents, desc="Loading documents"):
        num_documents += 1
        docid = document[id_field]
        docid_analyzed = f"{docid}__analyzed"
        if docid not in storage:
            storage[docid] = document
        if docid_analyzed not in storage:
            storage[docid_analyzed] = cast(Document, {"id": docid, "tokens": analyzer(document[text_field])})
        vocab.add_document(storage[docid_analyzed]["tokens"])

    storage.flush()

    def iter_analyzed_texts() -> Iterable[Tuple[str, Sequence[str]]]:
        assert storage is not None
        for docid, doc in storage.items():
            if docid.endswith("__analyzed"):
                docid = docid[:-10]
                yield docid, doc["tokens"]

    indexer: Union[InvertedIndexer, AnnSparseIndexer]
    if approximate_search:
        indexer = AnnSparseIndexer(**(indexer_config or {}))
    else:
        indexer = InvertedIndexer(**(indexer_config or {}))

    vectorizer = TfidfVectorizer(vocab)

    num_batches = math.ceil(num_documents / batch_size)
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
        indexer.build(print_progress=True, **(postprocessing_config or {}))

    return TinySearch(
        storage=storage,
        indexer=indexer,
        vectorizer=vectorizer,
        analyzer=analyzer,
    )


def sif(
    documents: Iterable[Document],
    embeddings: Union[str, PathLike, Mapping[str, DenseMatrix]],
    *,
    id_field: str = "id",
    text_field: str = "text",
    probabilities: Optional[Mapping[str, float]] = None,
    smoothing: float = 1e-3,
    batch_size: int = 1000,
    approximate_search: bool = False,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    indexer_config: Optional[Mapping[str, Any]] = None,
    postprocessing_config: Optional[Mapping[str, Any]] = None,
) -> TinySearch[Document, DenseMatrix]:

    from tinysearch.analyzers import WhitespaceTokenizer
    from tinysearch.indexers import AnnDenseIndexer, DenseIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import SifVectorizer
    from tinysearch.vocabulary import Vocabulary

    if isinstance(documents, Storage):
        storage = documents
    elif storage is None:
        storage = MemoryStorage()

    if isinstance(embeddings, (str, PathLike)):
        embeddings = util.spinner(desc=f"Loading embeddings from {embeddings}")(util.load_pretrained_embeddings)(
            embeddings
        )

    vocab = Vocabulary() if probabilities is None else None
    analyzer = analyzer or WhitespaceTokenizer()
    num_documents = 0
    for document in util.progressbar(documents, desc="Loading documents"):
        num_documents += 1
        docid = document[id_field]
        docid_analyzed = f"{docid}__analyzed"
        if docid not in storage:
            storage[docid] = document
        if docid_analyzed not in storage:
            storage[docid_analyzed] = cast(Document, {"id": docid, "tokens": analyzer(document[text_field])})
        if vocab is not None:
            vocab.add_document(storage[docid_analyzed]["tokens"])

    storage.flush()

    def iter_analyzed_texts() -> Iterable[Tuple[str, Sequence[str]]]:
        assert storage is not None
        for docid, doc in storage.items():
            if docid.endswith("__analyzed"):
                docid = docid[:-10]
                yield docid, doc["tokens"]

    if probabilities is None:
        assert vocab is not None
        probabilities = {token: vocab.get_token_probability(token) for token in vocab.token_to_index}

    indexer_config = {"space": "cosine", **(indexer_config or {})}

    indexer: Union[DenseIndexer, AnnDenseIndexer]
    if approximate_search:
        indexer = AnnDenseIndexer(**indexer_config)
    else:
        indexer = DenseIndexer(**indexer_config)

    vectorizer = SifVectorizer(probabilities, embeddings, smoothing=smoothing)

    num_batches = math.ceil(num_documents / batch_size)
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
        indexer.build(print_progress=True, **(postprocessing_config or {}))

    return TinySearch(
        storage=storage,
        indexer=indexer,
        vectorizer=vectorizer,
        analyzer=analyzer,
    )


def swem(
    documents: Iterable[Document],
    embeddings: Union[str, PathLike, Mapping[str, DenseMatrix]],
    *,
    id_field: str = "id",
    text_field: str = "text",
    window_size: int = 3,
    smoothing: float = 1e-3,
    batch_size: int = 1000,
    approximate_search: bool = False,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    indexer_config: Optional[Mapping[str, Any]] = None,
    postprocessing_config: Optional[Mapping[str, Any]] = None,
) -> TinySearch[Document, DenseMatrix]:

    from tinysearch.analyzers import WhitespaceTokenizer
    from tinysearch.indexers import AnnDenseIndexer, DenseIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import SwemVectorizer

    if isinstance(embeddings, (str, PathLike)):
        embeddings = util.spinner(desc=f"Loading embeddings from {embeddings}")(util.load_pretrained_embeddings)(
            embeddings
        )

    if isinstance(documents, Storage):
        storage = documents
    elif storage is None:
        storage = MemoryStorage()

    analyzer = analyzer or WhitespaceTokenizer()
    num_documents = 0
    for document in util.progressbar(documents, desc="Loading documents"):
        num_documents += 1
        docid = document[id_field]
        docid_analyzed = f"{docid}__analyzed"
        if docid not in storage:
            storage[docid] = document
        if docid_analyzed not in storage:
            storage[docid_analyzed] = cast(Document, {"id": docid, "tokens": analyzer(document[text_field])})

    storage.flush()

    def iter_analyzed_texts() -> Iterable[Tuple[str, Sequence[str]]]:
        assert storage is not None
        for docid, doc in storage.items():
            if docid.endswith("__analyzed"):
                docid = docid[:-10]
                yield docid, doc["tokens"]

    indexer_config = {"space": "cosine", **(indexer_config or {})}

    indexer: Union[DenseIndexer, AnnDenseIndexer]
    if approximate_search:
        indexer = AnnDenseIndexer(**indexer_config)
    else:
        indexer = DenseIndexer(**indexer_config)

    vectorizer = SwemVectorizer(embeddings, window_size=window_size, smoothing=smoothing)

    num_batches = math.ceil(num_documents / batch_size)
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
        indexer.build(print_progress=True, **(postprocessing_config or {}))

    return TinySearch(
        storage=storage,
        indexer=indexer,
        vectorizer=vectorizer,
        analyzer=analyzer,
    )


def transformer(
    documents: Iterable[Document],
    model_name: str,
    *,
    id_field: str = "id",
    text_field: str = "text",
    vectorization_method: str = "sum",
    batch_size: int = 1000,
    approximate_search: bool = False,
    storage: Optional[Storage[Document]] = None,
    analyzer: Optional[Callable[[str], str]] = None,
    indexer_config: Optional[Mapping[str, Any]] = None,
    postprocessing_config: Optional[Mapping[str, Any]] = None,
) -> TinySearch[Document, DenseMatrix]:

    from tinysearch.analyzers import PassThroughAnalyzer
    from tinysearch.indexers import AnnDenseIndexer, DenseIndexer
    from tinysearch.storages import MemoryStorage
    from tinysearch.vectorizers import TransformerVectorizer

    if isinstance(documents, Storage):
        storage = documents
    elif storage is None:
        storage = MemoryStorage()

    analyzer = analyzer or PassThroughAnalyzer()
    num_documents = 0
    for document in util.progressbar(documents, desc="Loading documents"):
        num_documents += 1
        docid = document[id_field]
        if docid not in storage:
            storage[docid] = document
        docid_analyzed = f"{docid}__analyzed"
        if docid_analyzed not in storage:
            storage[docid_analyzed] = cast(Document, {"id": docid, "tokens": analyzer(document[text_field])})

    storage.flush()

    def iter_analyzed_texts() -> Iterable[Tuple[str, Sequence[str]]]:
        assert storage is not None
        for docid, doc in storage.items():
            if docid.endswith("__analyzed"):
                docid = docid[:-10]
                yield docid, doc["tokens"]

    indexer_config = {"space": "cosine", **(indexer_config or {})}

    indexer: Union[DenseIndexer, AnnDenseIndexer]
    if approximate_search:
        indexer = AnnDenseIndexer(**indexer_config)
    else:
        indexer = DenseIndexer(**indexer_config)

    vectorizer = TransformerVectorizer(model_name, method=vectorization_method)

    num_batches = math.ceil(num_documents / batch_size)
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
        indexer.build(print_progress=True, **(postprocessing_config or {}))

    return TinySearch(
        storage=storage,
        indexer=indexer,
        vectorizer=vectorizer,
        analyzer=analyzer,
    )
