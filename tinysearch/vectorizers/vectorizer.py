import abc
from typing import Sequence

from scipy import sparse

from tinysearch.vocabulary import Vocabulary


class Vectorizer(abc.ABC):
    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab

    @abc.abstractmethod
    def vectorize_queries(self, queies: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        raise NotImplementedError

    @abc.abstractmethod
    def vectorize_documents(self, documents: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        raise NotImplementedError
