import abc
from typing import Generic, Sequence

from tinysearch.typing import Matrix
from tinysearch.vocabulary import Vocabulary


class Vectorizer(abc.ABC, Generic[Matrix]):
    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab

    @abc.abstractmethod
    def vectorize_queries(self, queies: Sequence[Sequence[str]]) -> Matrix:
        raise NotImplementedError

    @abc.abstractmethod
    def vectorize_documents(self, documents: Sequence[Sequence[str]]) -> Matrix:
        raise NotImplementedError
