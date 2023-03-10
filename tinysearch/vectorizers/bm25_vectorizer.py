import math
from collections import Counter
from typing import List, Sequence

from scipy import sparse

from tinysearch.vectorizers.vectorizer import Vectorizer
from tinysearch.vocabulary import Vocabulary


class BM25Vectorizer(Vectorizer[sparse.csr_matrix]):
    def __init__(self, vocab: Vocabulary, k1: float = 1.5, b: float = 0.75) -> None:
        self.vocab = vocab
        self.k1 = k1
        self.b = b

    def vectorize_queries(self, queies: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        row: List[int] = []
        col: List[int] = []
        data: List[float] = []
        for query_index, tokens in enumerate(queies):
            counter = Counter(tokens)
            for token, count in counter.items():
                if token not in self.vocab:
                    continue
                token_index = self.vocab[token]
                row.append(query_index)
                col.append(token_index)
                data.append(count)

        shape = (len(queies), len(self.vocab))
        return sparse.csr_matrix((data, (row, col)), shape=shape)

    def vectorize_documents(self, documents: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        k1 = self.k1
        b = self.b
        N = self.vocab.number_of_documents
        avgl = self.vocab.average_document_length

        row: List[int] = []
        col: List[int] = []
        data: List[float] = []
        for document_index, tokens in enumerate(documents):
            tokens = [token for token in tokens if token in self.vocab]
            counter = Counter(tokens)
            D = len(tokens)
            for token, count in counter.items():
                n = self.vocab.document_frequency[token]
                idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
                weight = count * (k1 + 1) / (count + k1 * (1 - b + b * D / avgl))
                value = idf * weight
                token_index = self.vocab[token]
                row.append(document_index)
                col.append(token_index)
                data.append(value)

        shape = (len(documents), len(self.vocab))
        return sparse.csr_matrix((data, (row, col)), shape=shape)
