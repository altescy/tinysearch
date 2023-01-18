import dataclasses
from collections import Counter
from typing import Dict, Iterable, Sequence, Type, TypeVar

Self = TypeVar("Self", bound="Vocabulary")


@dataclasses.dataclass(frozen=True)
class Vocabulary:
    token_to_index: Dict[str, int]
    index_to_token: Dict[int, str]
    token_frequency: Dict[str, int]
    document_frequency: Dict[str, int]
    number_of_tokens: int
    number_of_documents: int
    average_document_length: float

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

    @classmethod
    def from_documents(
        cls: Type[Self],
        documents: Iterable[Sequence[str]],
        stopwords: Sequence[str] = (),
    ) -> Self:
        token_to_index: Dict[str, int] = {}
        index_to_token: Dict[int, str] = {}
        token_frequency: Dict[str, int] = {}
        document_frequency: Dict[str, int] = {}
        number_of_tokens = 0
        number_of_documents: int = 0
        average_document_length: float = 0.0

        for tokens in documents:
            tokens = [token for token in tokens if token not in stopwords]
            counter = Counter(tokens)
            number_of_documents += 1
            number_of_tokens += sum(counter.values())
            average_document_length += len(tokens)
            for token, count in counter.items():
                if token not in token_to_index:
                    index = len(token_to_index)
                    token_to_index[token] = index
                    index_to_token[index] = token
                    token_frequency[token] = 0
                token_frequency[token] += count
                document_frequency[token] = document_frequency.get(token, 0) + 1

        average_document_length /= number_of_documents

        return cls(
            token_to_index=token_to_index,
            index_to_token=index_to_token,
            token_frequency=token_frequency,
            document_frequency=document_frequency,
            number_of_tokens=number_of_tokens,
            number_of_documents=number_of_documents,
            average_document_length=average_document_length,
        )
