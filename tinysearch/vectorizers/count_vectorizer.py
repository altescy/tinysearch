from collections import Counter
from typing import List, Sequence

from scipy import sparse

from tinysearch.vectorizers.vectorizer import Vectorizer
from tinysearch.vocabulary import Vocabulary


class CountVectorizer(Vectorizer):
    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)

    def _get_count_vectors(self, documents: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        row: List[int] = []
        col: List[int] = []
        data: List[int] = []
        for document_index, tokens in enumerate(documents):
            counter = Counter(tokens)
            for token, count in counter.items():
                if token not in self.vocab:
                    continue
                token_index = self.vocab[token]
                row.append(document_index)
                col.append(token_index)
                data.append(count)
        shape = (len(documents), len(self.vocab))
        return sparse.csr_matrix((data, (row, col)), shape=shape)

    def vectorize_queries(self, queries: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        return self._get_count_vectors(queries)

    def vectorize_documents(self, documents: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        return self._get_count_vectors(documents)
