from typing import Mapping, Sequence

import numpy

from tinysearch.typing import DenseMatrix
from tinysearch.vectorizers.vectorizer import Vectorizer
from tinysearch.vocabulary import Vocabulary


class SifVectorizer(Vectorizer[DenseMatrix]):
    def __init__(
        self,
        vocab: Vocabulary,
        embeddings: Mapping[str, DenseMatrix],
        smoothing: float = 1e-3,
    ) -> None:
        super().__init__(vocab)
        self.smoothing = smoothing
        self.embeddings = embeddings
        self.embedding_dim = len(next(iter(embeddings.values())))

    def _get_sif_vector(self, document: Sequence[str]) -> DenseMatrix:
        vector = numpy.zeros(self.embedding_dim)
        num_tokens = 0
        for token in document:
            if token not in self.vocab:
                continue
            token_probability = self.vocab.get_token_probability(token)
            vector += self.smoothing / (self.smoothing + token_probability) * self.embeddings[token]
            num_tokens += 1
        return vector / num_tokens

    def vectorize_queries(self, queies: Sequence[Sequence[str]]) -> DenseMatrix:
        return numpy.vstack([self._get_sif_vector(query) for query in queies])

    def vectorize_documents(self, documents: Sequence[Sequence[str]]) -> DenseMatrix:
        return numpy.vstack([self._get_sif_vector(document) for document in documents])
