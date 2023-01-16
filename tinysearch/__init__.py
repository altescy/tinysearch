from importlib.metadata import version
from typing import Callable, Iterable, Optional, Sequence

from tinysearch.indexers import Indexer, ScipyIndexer  # noqa: F401
from tinysearch.storages import MemoryStorage, Storage  # noqa: F401
from tinysearch.tinysearch import TinySearch
from tinysearch.typing import Document
from tinysearch.vectorizers import CountVectorizer, Vectorizer  # noqa: F401
from tinysearch.vocabulary import Vocabulary  # noqa: F401

__version__ = version("tinysearch")


def from_documents(
    documents: Iterable[Document],
    batch_size: int = 1000,
    storage: Optional[Storage[Document]] = None,
    indexer: Optional[Indexer] = None,
    vectorizer: Optional[Vectorizer] = None,
    analyzer: Optional[Callable[[str], Sequence[str]]] = None,
    stopwords: Optional[Sequence[str]] = None,
) -> TinySearch[Document]:
    return TinySearch.from_documents(
        documents,
        batch_size=batch_size,
        storage=storage,
        indexer=indexer,
        vectorizer=vectorizer,
        analyzer=analyzer,
        stopwords=stopwords,
    )
