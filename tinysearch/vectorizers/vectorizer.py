import abc
from typing import Generic, Sequence

from tinysearch.typing import Matrix


class Vectorizer(abc.ABC, Generic[Matrix]):
    @abc.abstractmethod
    def vectorize_queries(self, queies: Sequence[Sequence[str]]) -> Matrix:
        raise NotImplementedError

    @abc.abstractmethod
    def vectorize_documents(self, documents: Sequence[Sequence[str]]) -> Matrix:
        raise NotImplementedError
