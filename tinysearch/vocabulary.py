import dataclasses
from typing import Dict, Iterable, MutableMapping, Sequence, Type, TypeVar

Self = TypeVar("Self", bound="Vocabulary")


@dataclasses.dataclass(frozen=True)
class Vocabulary:
    token_to_index: MutableMapping[str, int]
    index_to_token: MutableMapping[int, str]
    document_frequency: MutableMapping[str, int]
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

    @classmethod
    def from_documents(
        cls: Type[Self],
        documents: Iterable[Sequence[str]],
        stopwords: Sequence[str] = (),
    ) -> Self:
        token_to_index: Dict[str, int] = {}
        index_to_token: Dict[int, str] = {}
        document_frequency: Dict[str, int] = {}
        number_of_documents: int = 0
        average_document_length: float = 0.0

        for tokens in documents:
            number_of_documents += 1
            average_document_length += len(tokens)
            for token in set(tokens):
                if token in stopwords:
                    continue
                if token not in token_to_index:
                    index = len(token_to_index)
                    token_to_index[token] = index
                    index_to_token[index] = token
                document_frequency[token] = document_frequency.get(token, 0) + 1

        average_document_length /= number_of_documents

        return cls(
            token_to_index=token_to_index,
            index_to_token=index_to_token,
            document_frequency=document_frequency,
            number_of_documents=number_of_documents,
            average_document_length=average_document_length,
        )
