import math
from collections import Counter
from typing import List, Sequence

from scipy import sparse

from tinysearch.vectorizers.vectorizer import Vectorizer
from tinysearch.vocabulary import Vocabulary


class TfidfVectorizer(Vectorizer[sparse.csr_matrix]):
    def __init__(self, vocab: Vocabulary) -> None:
        self.vocab = vocab

    def _get_tfidf_vectors(self, documents: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        row: List[int] = []
        col: List[int] = []
        data: List[float] = []
        for document_index, tokens in enumerate(documents):
            counter = Counter(tokens)
            for token, count in counter.items():
                if token not in self.vocab:
                    continue
                idf = 1.0 + math.log(
                    (1.0 + self.vocab.number_of_documents) / (1.0 + self.vocab.document_frequency[token])
                )
                token_index = self.vocab[token]
                token_value = count * idf
                row.append(document_index)
                col.append(token_index)
                data.append(token_value)
        shape = (len(documents), len(self.vocab))
        return sparse.csr_matrix((data, (row, col)), shape=shape)

    def vectorize_queries(self, queries: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        return self._get_tfidf_vectors(queries)

    def vectorize_documents(self, documents: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        return self._get_tfidf_vectors(documents)
