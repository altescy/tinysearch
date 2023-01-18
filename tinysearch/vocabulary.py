from collections import Counter
from typing import Dict, Iterable, Sequence, TypeVar

Self = TypeVar("Self", bound="Vocabulary")


class Vocabulary:
    def __init__(
        self,
        stopwords: Sequence[str] = (),
    ) -> None:
        self.token_to_index: Dict[str, int] = {}
        self.index_to_token: Dict[int, str] = {}
        self.token_frequency: Dict[str, int] = {}
        self.document_frequency: Dict[str, int] = {}
        self.number_of_tokens: int = 0
        self.number_of_documents: int = 0

    def __len__(self) -> int:
        return len(self.token_to_index)

    def __iter__(self) -> Iterable[str]:
        return iter(self.token_to_index)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_index

    def __getitem__(self, token: str) -> int:
        return self.token_to_index[token]

    def get_token_by_index(self, index: int) -> str:
        return self.index_to_token[index]

    def get_index_by_token(self, token: str) -> int:
        return self.token_to_index[token]

    def get_token_probability(self, token: str, smoothing: float = 1.0) -> float:
        return (self.token_frequency[token] + smoothing) / (self.number_of_tokens + smoothing * len(self))

    @property
    def average_document_length(self) -> float:
        return self.number_of_tokens / self.number_of_documents

    def add_document(self, document: Sequence[str]) -> None:
        self.number_of_documents += 1
        counter = Counter(document)
        for token, count in counter.items():
            if token not in self.token_to_index:
                index = len(self.token_to_index)
                self.token_to_index[token] = index
                self.index_to_token[index] = token
                self.token_frequency[token] = 0
                self.document_frequency[token] = 0
            self.token_frequency[token] += count
            self.document_frequency[token] += 1
        self.number_of_tokens += len(document)

    def add_documents(self, documents: Iterable[Sequence[str]]) -> None:
        for document in documents:
            self.add_document(document)
