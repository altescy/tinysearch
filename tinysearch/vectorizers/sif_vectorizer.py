from typing import Mapping, Sequence

import numpy

from tinysearch.typing import DenseMatrix
from tinysearch.vectorizers.vectorizer import Vectorizer


class SifVectorizer(Vectorizer[DenseMatrix]):
    def __init__(
        self,
        probabilities: Mapping[str, float],
        embeddings: Mapping[str, DenseMatrix],
        smoothing: float = 1e-3,
        normalize: bool = False,
    ) -> None:
        self.probabilities = probabilities
        self.smoothing = smoothing
        self.normalize = normalize
        self.embeddings = embeddings
        self.embedding_dim = len(next(iter(embeddings.values())))

    def _get_sif_vector(self, document: Sequence[str]) -> DenseMatrix:
        vector = numpy.zeros(self.embedding_dim)
        num_tokens = 0
        for token in document:
            if token not in self.probabilities:
                continue
            token_probability = self.probabilities[token]
            weight = max(1e-8, self.smoothing / (self.smoothing + token_probability))
            vector += weight * self.embeddings[token]
            num_tokens += 1
        if num_tokens != 0:
            vector = vector / num_tokens
        if self.normalize:
            vector = vector / (numpy.linalg.norm(vector) + 1e-8)
        return vector

    def vectorize_queries(self, queies: Sequence[Sequence[str]]) -> DenseMatrix:
        return numpy.vstack([self._get_sif_vector(query) for query in queies])

    def vectorize_documents(self, documents: Sequence[Sequence[str]]) -> DenseMatrix:
        return numpy.vstack([self._get_sif_vector(document) for document in documents])
